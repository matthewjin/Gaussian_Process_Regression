function [k s_n] = index_compute_kernel_parameters(sampling_matrix, prior_matrix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute kernel parameters based on known sample values and prior belief
%on mean (determined via LOWESS).	
% INPUT:
% x: vector of x values (nx1)
% y: vector of y values (nx1)
% mu: vector of prior mean values (nx1)
% OUTPUT:
% k: function handle for prior kernel loaded with grid-search-optimized parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s_y_list=0.2:.1:1;
l_list=0.1:.1:1.2;
s_n_list= 0; 
n = size(sampling_matrix, 1);

% Set up Sigma, y, and prior_y
x = sampling_matrix(:, 1);
y = sampling_matrix(:, 2);
prior_y = NaN(n, 1);

for i = 1:n
  prior_y(i) = prior_matrix(prior_matrix(:, 1) == x(i), 2);
end

max_likelihood=-Inf;
max_s_y = NaN; max_l = NaN; max_s_n = NaN;
likelihood = @(Sigma) (-1/2)*log(det(Sigma))-((1/2)*(y-prior_y).')*(Sigma\(y-prior_y));

for s_y = s_y_list
  for l = l_list
    for s_n = s_n_list
      % Kernel params to test
      k = @(x_1, x_2) noiseless_kernel(s_y, l, x_1, x_2);
      
      % Compute likelihood
      Sigma = bsxfun(k, x, x') + s_n^2*eye(n);
      current_likelihood = likelihood(Sigma);
      
      disp([current_likelihood s_y l s_n])
      
      if current_likelihood > max_likelihood
	max_likelihood = current_likelihood;
	max_s_y = s_y;
	max_l = l;
	max_s_n = s_n;
      end
    end
  end
end

% Set the final kernel
k = @(x_1, x_2) kernel(max_s_y, max_l, x_1, x_2);
keyboard;
end