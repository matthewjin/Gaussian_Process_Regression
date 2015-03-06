function Sigma = compute_covariance_matrix(theta, x)
% Compute the covariance matrix based on kernel parameters and current
% values of x, where theta = [s_y, s_n, l]

s_y = theta(1); s_n = theta(2); l = theta(3);
step_0 = bsxfun(@minus, x', x).^2;
step_1 = exp(-step_0/(2*l^2));
step_2 = s_n^2*eye(length(x));
Sigma = s_y^2*step_1 + step_2;
    ;




