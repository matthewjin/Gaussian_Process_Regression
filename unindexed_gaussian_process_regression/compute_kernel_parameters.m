function k = compute_kernel_parameters(x,y,mu)
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
s_n_list=0; %0.2:.1:1;

max_likelihood=-Inf;
max_s_y = NaN; max_l = NaN; max_s_n = NaN;

likelihood = @(Sigma) (-1/2)*log(det(Sigma))-((1/2)*(y-mu).')*(Sigma\(y-mu));

for s_y = s_y_list
  for l = l_list
    for s_n = s_n_list
      % Kernel params to test
      k = @(x_1, x_2) kernel(s_y, l, s_n, x_1, x_2);
      
      % Compute likelihood
      Sigma = bsxfun(k, x, x');
      current_likelihood = likelihood(Sigma);
      
      if current_likelihood > max_likelihood
	max_s_y = s_y;
	max_l = l;
	max_s_n = s_n;
      end
    end
  end
end

% Set the final kernel
k = @(x_1, x_2) kernel(max_s_y, max_l, max_s_n, x_1, x_2);
end