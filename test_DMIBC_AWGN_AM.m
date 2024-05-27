% DMIBC     AWGN channel    AM algorithm
clear;clc;close all

%% input
SNRdB = 10;     % signal-to-noise ratio
SNR = 10 .^ (SNRdB ./ 10);        sigma = 1 / sqrt(2 * SNR);  % parameter in normal distribution
eta = 0.9;  theta = pi/18;      % parameters of the AWGN channel

m0 = 4;      M = m0^2;    % number of discrete points for the alphabet X
n0 = 50;     N = n0^2;    % number of discrete points for the alphabet Y, Z

% alphabet
% set X
xI = zeros(1,M);    xQ = zeros(1,M);
for i = 1 : m0
    for j = 1 : m0
        xI((i-1) * m0 + j) = (m0-1) - (i-1) * 2;    
        xQ((i-1) * m0 + j) = (m0-1) - (j-1) * 2;
    end
end
xdelta = (sum(xI.*xI) + sum(xQ.*xQ)) / M;   xI = xI ./ sqrt(xdelta);      xQ = xQ ./ sqrt(xdelta);
xn2 = (xI .^ 2 + xQ .^ 2)';     %\|x_i\|_^2
% HX 
HxI = zeros(1,M);   HxQ = zeros(1,M);
for i = 1:M
    HxI(i) = xI(i)*cos(theta) + xQ(i)*sin(theta);
    HxQ(i) = -xI(i)*sin(theta) + xQ(i)*cos(theta);
end
HxQ = eta * HxQ;
% position Y = Z
yint = -8;      y_delta= 2*(-yint) / (n0-1);
yI = zeros(1,N);    yQ = zeros(1,N); 
for k = 1:N
    r = floor((k-1)/sqrt(N));   s = mod(k-1, sqrt(N))+1;
    yI(k) = yint + r * y_delta;
    yQ(k) = yint + (s-1) * y_delta;
end
 
% decoding metric  d(x_i,z_j)   m*n
d = (yI-xI') .^ 2 + (yQ-xQ') .^ 2;

% transition law s_{ij} = W(y_j|x_i) = F(y_j - Hx_i)    m*k
s = zeros(M,N);
for j = 1:N
    for i = 1:M
        py1 = normcdf(yI(j)+y_delta/2,HxI(i),sigma) - normcdf(yI(j)-y_delta/2,HxI(i),sigma);
        py2 = normcdf(yQ(j)+y_delta/2,HxQ(i),sigma) - normcdf(yQ(j)-y_delta/2,HxQ(i),sigma);
        s(i,j) = py1 * py2;
    end
end


%% AM algorithm
max_iter = 5000;  
lambda = 0.5;    % fixed IB Lagrange multiplier
Gam = 1;    % power constraint

% opt_lm: I_{LM}(X;Z).  opt_bot: I(Y;Z). r_stop: max(residual errors).
% p_out: optimized input distribution P_X. 
% t: optimized conditional distribution P_{Z|Y}
op_ans_time = tic;
[opt_lm, opt_bot, r_stop, p_out, opttt1, opttt2, zeta, mu, r_phi, r_psi, r_zeta, r_mu, t] = DMIBC_AM(d, s, xn2, Gam, lambda, max_iter);
op_ans_time = toc(op_ans_time);
p_shape = fliplr(reshape(p_out',[m0,m0]));  


% %% Convergent trajectories of the residual errors 
% r_phi(r_phi<1e-18) = 1e-18;     r_psi(r_psi<1e-18) = 1e-18; 
% figure(1)
% semilogy([1:1:length(r_phi)],r_phi,'color',[0.9290 0.6940 0.1250],'linewidth',1)
% hold on
% semilogy([1:1:length(r_psi)],r_psi,'b','linewidth',1)
% hold on
% semilogy([1:1:length(r_zeta)],r_zeta,'color',[0.4940 0.1840 0.5560],'linewidth',1)
% hold on
% semilogy([1:1:length(r_mu)],r_mu,'--g','linewidth',1)
% hold off
% xlabel('iteration steps')
% ylabel('residual error')
% legend('$r_{\phi}$','$r_{\psi}$','$r_{\zeta}$', '$r_{\mu}$')
% hold off

