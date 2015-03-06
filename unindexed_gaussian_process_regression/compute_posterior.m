function posterior_matrix = compute_posterior(k, prior_matrix, x, y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the posterior gaussian distribution and plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find all prior-belief output values for known samples
prior_y_index = ismember(prior_matrix(:, 1), x);
prior_y = prior_matrix(prior_y_index, 2);

% Compute the Sigma matrix and prior_Sigma matrix
Sigma = bsxfun(k, x, x');
inv_Sigma = inv(Sigma);
prior_Sigma = bsxfun(k, x, prior_matrix(:,1)');

% Compute posterior:
posterior_matrix = prior_matrix; % Create posterior matrix

for i = 1 : size(prior_matrix, 1)
  % Compute the posterior mean
  posterior_matrix(i, 2) = prior_matrix(i, 2) + ...
      prior_Sigma(:, i)'*inv_Sigma*(y-prior_y);
  % Compute the posterior variance
  posterior_matrix(i, 3) = k(prior_matrix(i, 1), prior_matrix(i, 1)) ...
      - prior_Sigma(:, i)'*inv_Sigma*prior_Sigma(:, i);
end

% Set negative values to zero
posterior_matrix(posterior_matrix(:, 3) < 0, 3) = 0;