% DMIBC    Quaternary Channel MMIB algorithm
clear;clc;close all

%% input
B = 0.81;   % the Bottleneck constraint B = I(Y;Z).
B0 = 4 * (B - 2); % I(Y;Z) = B, i.e., sum(sum(t .* log2(t))) - B0 = 0.

ep = 0.3;   % parameter of the channel
% decoding metric  d(x_i,z_j)
qd = [1-ep, ep/3, ep/3, ep/3; ep/3, 1-ep, ep/3, ep/3; ep/3, ep/3, 1-ep, ep/3; ep/3, ep/3, ep/3, 1-ep];
d = -log(qd);
% transition law s_{ij}
s = [1-ep, ep/2, 0, ep/2; ep/2, 1-ep, ep/2, 0; 0, ep/2, 1-ep, ep/2; ep/2, 0, ep/2, 1-ep];

% input distribution
p = [0.25; 0.25; 0.25; 0.25];   % P_X
q = s' * p;         %P_Y

%% MMIB algorithm

% Initial guess
k = 4; 
rand('seed', 1600);   A_center = rand(k);
A_org = A_center;   num_neib = 5;   eps_neib = 1;
iter_out = 0;   res_opt = 10;   flag = 0;
op_time = 0;  op_time_qualt = tic; 
opt_cur = 0;    % optimized LM rate

% Grid search
while iter_out <= 80  &&  flag + res_opt>1e-5
    iter_out = iter_out + 1;
    
    % Step 1: find the neighbors of A_center without constraint
    matrix_neib = findneib(A_center, eps_neib, num_neib);
    eps_neib = 0.8 * eps_neib; % adaptive grid seach parameter
    
    % Step 2: For each neighbor, find t=P{Z|Y} that satisfies the Bottleneck constraint B = I(Y;Z), and compute the LM rate. 
    for i = 1 : num_neib
        A_org = matrix_neib{i};
        % Step 2-1: solve I(Y;Z) = B to find P(Z|Y)
        [A_solution, res, iter_inner] = solpzy(A_org, B0, 1e-8);
        t = A_solution;
        t(t < 0) = 0;
        
        % Step 2-2:  Compute LM rate under the above t=P{Z|Y} using CVX
        r = t' * s' * p; 
        
        % compute the current I(Y;Z) for accuracy
        lgggg1 = log(t ./ r');   lgggg1(lgggg1<-1000) = -1000;
        opttt1 =  q' * sum(t .* lgggg1, 2)/log(2);    
        % do not meet the Bottleneck constraint. Discount this t=P{Z|Y}
        if B - opttt1 < 0
            continue
        end
        
        % compute the LM rate
        op_temp = tic;
        T = (1/k) * sum(sum(s * t .* d)); %\sum\sum p_i P(z|x)logd(x,z)       
        [opt_cvx] = lm_cvx(p, r, d, T);
        % if : a higher LM rate
        if opt_cur < opt_cvx && opt_cvx < 10 && opt_cvx > 1e-5
            flag = 0;
            res_opt = opt_cvx - opt_cur;   % Numerical Gain of the LM rate
            A_center = A_solution;        % new initial point of grid search 
            opt_cur = opt_cvx;     % max value of LM rate before the current stage
            t_cur = t;     %t = P_{Z|Y} in current stage
        end 
        op_temp= toc(op_temp);
        % time for computing the LM rate
        op_time = op_temp + op_time;
    end
    % the end of compute LM rate for every neighbor
    
end
% the end of grid search

op_time_qualt = toc(op_time_qualt);


%% optimal solution
% opt_cur: I_{LM}(X;Z)
% t: optimized conditional distribution P_{Z|Y}
t = t_cur;
% r: distribution P_Z
r = t' * s' * p; 

% I(Y;Z) for accuracy
lgggg1 = log(t ./ r');   lgggg1(lgggg1<-1000) = -1000;
opttt1 =  q' * sum(t .* lgggg1, 2)/log(2);
% I(X;Z)
lgggg2 = log((s*t) ./ r');   lgggg2(lgggg2<-1000) = -1000;
opttt2 =  p' * sum((s*t) .* lgggg2, 2)/log(2);

res_bot = B - opttt1;
ans_MMIB = [ep, opt_cur, opttt1, opttt2, op_time, op_time_qualt, res_opt];

