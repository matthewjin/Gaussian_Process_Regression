function output = noiseless_kernel(s_y, l, x_1, x_2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kernel function: given x_1, x_2, outputs covariance 

%The current kernel
%output assumes that we only sample at each location once. Since we replace
%kronecker notation with x_1 == x_2 notation. Modification will be needed
%for noisy GP.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output = s_y^2*exp(-0.5*(x_1-x_2).^2/l^2);




