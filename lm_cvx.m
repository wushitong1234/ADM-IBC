% compute LM rate using cvx
% p  marginal distribution of X   M*1
% r  marginal distribution of Z   N*1
% d  decoding metric   m*n
% T constant in the constraint \sum\sum p_i P(z|x)logd(x,z)  
% C constant in the objective function \sum p log(p) + \sum q log(q)

function [opt_cvx] = lm_cvx(p, r, d, T)
    [M,~] = size(p);    [N,~] = size(r);
    C1 = sum(p.*log(p));    C2 = 0;
    for j = 1:N
        if r(j)*log(r(j)) < 0
            C2 = C2 + r(j)*log(r(j));
        end
    end
    C = C1 + C2;

    %% for cvx
    run('D:\cvx-w64\cvx\cvx_startup.m')
    n = M*N; 
    % tic;
    % cvx_solver sedumi
    cvx_begin 
        variable x(n)
        minimize( sum(rel_entr(x,1)) )
        subject to
            (1e-500) .* ones(n,1) <= x
            sum(sum(reshape(x,M,N) .* d)) <= T
            reshape(x,M,N) * ones(N,1) == p    % GMI no
            ones(1,M) * reshape(x,M,N) == r'      
    cvx_end
    % toc;

    opt_cvx = (cvx_optval - C)/log(2);
end
