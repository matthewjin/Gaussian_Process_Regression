function [min_theta, min_nll] = rui_fmingd(handle, init_theta, tol, max_iter)

theta = init_theta;
[nll grad] = handle(theta);
prev = nll;
diff = 1;
count = 1;
min_nll = nll;

while (count < max_iter) && (diff > tol)
  theta = theta + 0.001*grad;
  [nll grad] = handle(theta);
  diff = abs(nll - prev);
  prev = nll;
  count = count + 1;

  if nll < min_nll
    min_nll = nll;
    min_theta = theta;
  end
end
