function outputs = hidden_function(inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For given array of inputs, compute the corresponding array of outputs:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputs = noiseless_GP(inputs);

function outputs = simple(inputs)
outputs = inputs+2;

function outputs = screwy(inputs)
outputs = exp(sqrt(abs(inputs))).*sin(inputs);

function outputs = sine_wave(inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For given array of inputs, compute the corresponding array of outputs for
% sine wave:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputs = sin(inputs);

function outputs = noiseless_GP(inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noiseless gaussian process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xs = inputs;
ns = size(xs,1); keps = 1e-9;
m = inline('2*x.^.5');
K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
outputs = m(xs) + chol(K(xs,xs)+keps*eye(ns))'*randn(ns,1) ;

function outputs = noisy_GP(inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noisy gaussian process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xs = inputs;
ns = size(xs,1); keps = 1e-9;
m = inline('2*x.^.5');
K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
%outputs = m(xs) + chol(ones(ns, ns))'*randn(ns,1) ;
outputs = m(xs) + chol(K(xs,xs)+keps*eye(ns))'*randn(ns,1) + 0.2*randn(ns, 1);
