function posterior_matrix = establish_posterior_matrix(range, full_size)

posterior_matrix = [linspace(range(1), range(2), full_size)', zeros(full_size, 1), ...
      zeros(full_size, 1)];

