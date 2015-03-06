function main()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main function: tests gaussian process regression. Offline case at the
%moment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings
scale = 2;		% For scaling the variance (visual)
range = [1, 50];
full_size = 1000;
init_size = 10;
rng(100);		% Set seed

% Establish a posterior_matrix that will be updated
posterior_matrix = [linspace(range(1), range(2), full_size)', zeros(full_size, 1), ...
  zeros(full_size, 1)];

% Generate data using noiseless GP
hidden_x = posterior_matrix(:, 1);	% Generate a realization of the GP
hidden_y = hidden_function(hidden_x);	% For all values of x in full_size
x_index = round(linspace(1,full_size,init_size));     
x = hidden_x(x_index);			% Select init sampling 
y = hidden_y(x_index);		

% Establish prior gaussian process with LOWESS
[~,~,~, xy] = lowess([x y],1,0,0,hidden_x);    % Establish prior mean
posterior_matrix(:, 2) = xy(:, 2);
muy=xy(:,2);

k = computeKernelParam(x,y,muy(x_index)); % Establish prior kernel

% Compute the posterior gaussian process
count = 1; % Set counter
posterior_matrix = compute_posterior(k, posterior_matrix, x, y);
plot_gp(posterior_matrix, x, y, hidden_x, hidden_y, NaN, scale);
pause(1);

% GAUSSIAN UPDATE PROCESS
while true
  %Find location of max variance
  index = find(posterior_matrix(:, 3) == max(posterior_matrix(:, 3)), 1);
  if ismember(hidden_x(index), x) || posterior_matrix(index, 3) < 1e-5
    fprintf('Number of iterations: %d\n', count);
    plot_gp(posterior_matrix, x, y, hidden_x, hidden_y, NaN, scale);
    return
  end
  
  % Increment
  count = count + 1; 

  % For some reason, it was very important that the rows are sorted...?
  xy = sortrows([[x; hidden_x(index)] [y; hidden_y(index)]], 1);
  x = xy(:, 1);
  y = xy(:, 2);
  
  % Establish prior gaussian process with LOWESS
  % Establish prior mean
  [~,~,~, xy] = lowess([x y],1,0,0,hidden_x);
  posterior_matrix(:, 2) = xy(:, 2);

  % Establish prior kernel
  % Find all prior-belief output values for known samples
  prior_y = posterior_matrix(ismember(posterior_matrix(:, 1), x), 2);
  k =computeKernelParam(x,y,prior_y); % kernel handle

  % Compute posterior gaussian process
  posterior_matrix = compute_posterior(k, posterior_matrix, x, y);
  
  % Pause and plot
  plot_gp(posterior_matrix, x, y, hidden_x, hidden_y, index, scale);
  pause(1);
end

function plot_gp(posterior_matrix, x, y, hidden_x, hidden_y, index, scale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot the gaussian process posterior, along with the true function, as well
%as the confidence intervals.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf;
hold on;
% Confidence Intervals
upper = posterior_matrix(:, 2) + scale*sqrt(posterior_matrix(:, 3));
lower = posterior_matrix(:, 2) - scale*sqrt(posterior_matrix(:, 3));
X = [posterior_matrix(:, 1); fliplr(posterior_matrix(:, 1))];
Y = [upper; fliplr(lower)];
f = fill(X, Y, '--y');
set(f,'edgecolor','none');
% True function
plot(hidden_x, hidden_y, '-b')
% Real sampled points from true function
scatter(x, y);
% Interpolation
plot(posterior_matrix(:, 1), posterior_matrix(:, 2), '--k');
% Red dot for added point (in update loop)
if ~isnan(index)
  plot(hidden_x(index), hidden_y(index), 'ro', 'MarkerFaceColor', 'r',...
      'MarkerSize', 10);
end

title('Regression On a Fixed Realization of a Gaussian Process')
xlabel('x');
ylabel('f(x)')
legend('95% Confidence Interval','True Function','Sampled Points','Interpolation','Location','NorthWest');

