function  [s_y, s_n, l] = compute_kernel_parameter(mu, x, y)
%
% Compute the kernel parameters for noise-accoutned square exponential
% kernel, where theta = [s_y, s_n, l]
%

% Settings
init_theta = abs(randn(3, 1));
handle = @(theta) neg_log_likelihood(theta, mu, x, y);
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 1000);
% Run fminunc to obtain the optimal theta
% This function will return theta and the cost 
[theta, cost] = ...
    fminunc(handle, init_theta, options);

theta
cost

keyboard

%{
%handle = @(theta) tester(theta, mu, x, y);
[theta, cost] = ...
    rui_fmingd(handle, init_theta, 1e-6, 1000);
options = optimset('MaxIter', 50);
[theta cost] = fmincg(handle, init_theta, options);
[a b] = handle([1 1 1])
%}

function [nll grad] = neg_log_likelihood(theta, mu, x, y)
%
% Compute the log likelihood at a given value of theta
%

n = length(x); s_y = theta(1); s_n = theta(2); l = theta(3);
Sigma = compute_covariance_matrix(theta, x);
inv_Sigma = inv(Sigma);

% nll value
ll = -0.5*log(det(Sigma)) - 0.5*(y - mu)'*inv_Sigma*(y-mu);
nll = -1*ll;

% Compute grad_1: s_y
cross_sub = bsxfun(@minus, x', x);
S_sy = 2*s_y*exp(-cross_sub.^2/(2*l^2));
ll_grad_1 = 0.5*trace(inv_Sigma*S_sy) + 0.5*(y-mu)'*S_sy*inv_Sigma*S_sy*(y-mu);

% Compute grad_2: s_n
S_sn = 2*s_n*eye(n);
ll_grad_2 = 0.5*trace(inv_Sigma*S_sn) + 0.5*(y-mu)'*S_sn*inv_Sigma*S_sn*(y-mu);

% Compute grad_3: l
S_l = s_y^2 * cross_sub.^2/(l^3) .* exp(-cross_sub.^2/(2*l^2));
ll_grad_3 = 0.5*trace(inv_Sigma*S_l) + 0.5*(y-mu)'*S_l*inv_Sigma*S_l*(y-mu);

% Produce gradient
ll_grad = [ll_grad_1 ; ll_grad_2 ; ll_grad_3];
grad = -ll_grad;

function [value grad] = tester(theta, mu, x, y)
%
% A unit test
%
x = theta(1); y = theta(2); z = theta(3);
value = x^4 + y^4 + z^4 - 2*x*y*z;

grad_x = 4*x^3 - 2*y*z;
grad_y = 4*y^3 - 2*x*z;
grad_z = 4*z^3 - 2*x*y;

grad = [grad_x ; grad_y ; grad_z];
