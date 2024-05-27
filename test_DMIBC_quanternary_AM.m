% DMIBC    Quaternary Channel    AM algorithm with binary search
clear;clc;close all

%% input
B = 0.81;   % the Bottleneck constraint B = I(Y;Z)
ep = 0.3;   % parameter of the channel
% decoding metric  d(x_i,z_j)
qd = [1-ep, ep/3, ep/3, ep/3; ep/3, 1-ep, ep/3, ep/3; ep/3, ep/3, 1-ep, ep/3; ep/3, ep/3, ep/3, 1-ep];
d = -log(qd);
% transition law s_{ij}
s = [1-ep, ep/2, 0, ep/2; ep/2, 1-ep, ep/2, 0; 0, ep/2, 1-ep, ep/2; ep/2, 0, ep/2, 1-ep];

% input distribution
p = [0.25; 0.25; 0.25; 0.25];   % P_X
q = s' * p;         %P_Y

%% AM algorithm with binary search
lam_l = 0;   lam_r = 5;
iter_out = 0;   flag = 0;   opt_lm_old = 0;   opt_lm_cur = 100;

op_time_half = tic;
% main process of binary search
while iter_out <= 1000  && flag == 0
    iter_out = iter_out + 1
    opt_lm_old = opt_lm_cur;
    lambda = (lam_l + lam_r)/2;
    
    % Compute DM-IBC
    [opt_lm_cur, opt_bot, r_stop9, p_out9, optt1, opttt2, zeta, mu, r_phi, r_psi, r_zeta, r_mu, t] = DMIBC_AM_uni(d, s, lambda, 1e4);
    if opt_bot > B
        lam_l = lambda;
    else
        lam_r = lambda;
    end
    if abs(opt_lm_old - opt_lm_cur) < 1e-5  && (B - opt_bot) < 1e-9 && (B - opt_bot) >= 0 
        flag = 1;
    end
end
op_time_half = toc(op_time_half);

%% output
% opt_lm_cur: I_{LM}(X;Z).
% t: optimized conditional distribution P_{Z|Y}
% r: P_Z
r = t' * s' * p; 
% result
res_bot = B - opt_bot;
ans_AM = [lambda, opt_lm_cur, op_time_half, iter_out];

