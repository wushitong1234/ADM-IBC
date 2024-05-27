% DMIBC Alternating Maximization (AM) Algorithm  % with uniform input distribution
%% output
% opt_lm: I_{LM}(X;Z).  opt_bot: I(Y;Z).  r_stop: max(residual errors).
% p_out = p_input preset to the uniform distribution
% t: optimized conditional distribution P_{Z|Y}

%% input
% d: decoding metric  d(x_i,z_j)   m*n. 
% s: transition law s_{ij} = W(y_j|x_i) = F(y_j - Hx_i)    m*k 
% lambda: fixed Lagrange multiplier.
% max_iter: max iteration of AM algorithm

%% main function of DMIBC_AM_uni without optimizing the input distribution P_X
function [opt_lm, opt_bot, r_stop, p_out, opttt1, opttt2, zeta, mu, r_phi, r_psi, r_zeta, r_mu, t] = DMIBC_AM_uni(d, s, lambda, max_iter)
     % known variables
    [m,n] = size(d);    % d decoding metric   m*n
    [~,k] = size(s);	% s transition law    m*k
    nt_iter1 = 20;   % nt_iter2 = 20;     % max iteration times of Newton's method
    p = (1/m)* ones(m,1);	% marginal distribution of X   m*1
    q = s' * p;     q(q<1e-100) = 1e-100;   % marginal distribution of Y   k*1  
    
    % unknowm variables    
    % r: marginal distribution of Z   n*1
    rand('seed', 20);     r = rand(n, 1)+1;   r = r ./ sum(r);
    % t: conditional distribution Z|Y  k*n
    rand('seed', 23);     t = rand(k, n)+1;   t = t ./ sum(t,2);  % t_{kj} = P(z_j | y_k)
      
    % dual variables
    phi = ones(m,1);    psi = ones(n,1);
    zeta = 1;       K_ze = exp(-zeta*d);    mu = 1;
    r_phi = [];     r_psi = [];     r_zeta = [];     r_mu = [];
    iter = 0;       r_stop = 1000;

    % main iteration process
    while (iter < max_iter) && (r_stop > 1e-10)
        iter = iter + 1;
        r_mu(iter) = 2e-18;
        % compute the function $\bm J(\bdphi, \widetilde{\bdpsi}, \zeta; \bm \Omega, \bm r)$ 
        % tp: sum of the coefficients of linear terms related to the variable $p_i$    
        tp1 = (phi' * K_ze * (psi .* (s*t)'))';    %\sum_k\sum_j [\phi K_ze \psi s]
        tp2 = (s*t) * log(psi);     %\sum_j [s_ij *log(\psi_j)]
        tp3 = sum(d .* (s*t), 2);      %\sum_j [s_ij *d_ij]
        lg = log(t ./ r');   lg(lg<-1000) = -1000;
        tp4 = sum(s * (t .* lg), 2);
        tp = phi .* exp(-tp1 + tp2 - zeta * tp3 - lambda * tp4);   %T_i

        % Step 1: update P_{Z|Y}, P_Z (t, r)        
        tt1 = K_ze' * phi .* psi;
        tt2 = ((p .* s)' * d )./ q;
        ttt1 = exp((1/lambda)*(-tt1));
        ttt2 = exp((1/lambda)*(log(psi)));  ttt2(ttt2>1000) = 1000;
        ttt3 = exp(-(zeta/lambda)*tt2');
        t_2 = (r .* (ttt1.*ttt2).*ttt3)';
        t = t_2 ./ sum(t_2,2);   
        
        % compute P_z -r
        r = t' * s' * p;   r(r<1e-100) = 1e-100;   % r = r/sum(r);

        % Step 2:  compute the LM rate (phi psi zeta)
        phi = p ./ (K_ze * (psi .* r));     % update \phi
        psi = ones(n,1) ./ (K_ze' * phi);   % update \psi
        % update \zeta   
        % Solve G(\zeta) = 0 with Newton's method
        temp1 = Gfun(0,p,t,s,d,phi,psi);
        if temp1 <= 0
            x2 = 0;
            r_zeta(iter) = 2e-18;
        else
            r_zeta(iter) = abs(Gfun(zeta,p,t,s,d,phi,psi));
            x1 = zeta;
            x2 = x1-Gfun(x1,p,t,s,d,phi,psi)/Gfun_d(x1,r,d,phi,psi);
            t1 = 0; % iter of Newton's method 1
                while (abs(x2-x1) > 1e-10) && (t1 < nt_iter1)
                    t1 = t1+1;
                    x1 = x2;
                    x2 = x1-Gfun(x1,p,t,s,d,phi,psi)/Gfun_d(x1,r,d,phi,psi);
                end
        end
        zeta_old = zeta;    zeta = x2;  
        zeta(zeta<0) = 0;   zeta(find(isnan(zeta))) = 1;
        K_ze = exp(-zeta*d);   K_ze(K_ze<1e-100) = 1e-100;

        % residual errors 
        rr_phi = phi .* (K_ze * (psi .* r)) - p;
        r_phi(iter) = norm(rr_phi,1);
        rr_psi = (psi .* (K_ze' * phi) - ones(n,1)).*r;
        r_psi(iter) = norm(rr_psi,1);
        r_stop = max([r_zeta(iter), r_phi(iter), r_psi(iter), r_mu(iter)]);
     end
    % the end of main iteration process
    
    % compute LM rate between X and Z  -opt_lm
    lgp = log(p);       lgp(lgp<-1000) = -1000;
    lgpsi = log(psi);     lgpsi(lgpsi<-1000) = -1000;
    lgphi = log(phi);     lgphi(lgphi<-1000) = -1000;
    opt_y = - phi' * K_ze *(psi .* r) - p'*lgp + p' * lgphi + r' * lgpsi + 1 - zeta * sum(p' *(d .* (s*t)));
    opt_lm = opt_y/log(2);
    
    % compute the mutual information between Y and Z  -opt_bot
    lg = log(t ./ r');   lg(lg<-1000) = -1000;
    tp4 = sum(s * (t .* lg), 2);
    opt_x = p' * tp4;
    opt_bot = opt_x/log(2);
    p_out = p';
    
    % I(X;Y)
    lgggg1 = log(s ./ q');   lgggg1(lgggg1<-1000) = -1000;
    opttt1 =  p' * sum(s .* lgggg1, 2)/log(2);
    % I(X;Z)
    lgggg2 = log((s*t) ./ r');   lgggg2(lgggg2<-1000) = -1000;
    opttt2 =  p' * sum((s*t) .* lgggg2, 2)/log(2);
end

%% first order condition
% G(lambda)
function ww1 = Gfun(x,p,t,s,d,phi,psi)
    Ku = exp(-x*d);
    r = t' * s' * p;
    ww1 = phi' * (d .* Ku) * (psi .* r) - sum(p' *(d .* (s*t)));
end
% G^'(lambda)
function ww2 = Gfun_d(x,r,d,phi,psi)
    Ku = exp(-x*d);
    ww2 = - phi' * (d .* d .* Ku) * (psi .*r);
end

