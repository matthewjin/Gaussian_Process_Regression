function main()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main function: tests gaussian process regression. Offline case at the
%moment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings
addpath ./statistics;	% Add folder for LOWESS
scale = 2;		% For scaling the variance (visual)
range = [1, 4];
full_size = 1000;
init_size = 10; 
rng(100);		% Set seed
s_n = 0.1;

% Establish a posterior_matrix and sample that will be updated
posterior_matrix = establish_posterior_matrix(range, full_size);
sampling_matrix = establish_sampling_matrix(init_size);

% Construct the noisy_function (GP + noise)
hidden_x = posterior_matrix(:, 1);	% Generate a realization of the GP
hidden_y = hidden_function(hidden_x);	% For all values of x in full_size
noisy_function = @(x) make_noisy([hidden_x hidden_y], s_n, x);

% Sample from noisy function
sample_x = hidden_x(round(linspace(1,full_size,init_size)));
sample_y = noisy_function(sample_x);
sampling_matrix = [sample_x sample_y];

% Establish prior gaussian process with LOWESS
[~,~,~, xy] = lowess(sampling_matrix,1,0,0,hidden_x);    % Establish prior mean
posterior_matrix(:, 2) = xy(:, 2);

k = index_compute_kernel_parameters(sampling_matrix,posterior_matrix); % Establish prior kernel

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
  k = compute_kernel_parameters(x,y,prior_y); % kernel handle

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

