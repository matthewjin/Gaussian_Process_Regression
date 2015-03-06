% INPUT:
% x: vector of x values (nx1)
% y: vector of y values (nx1)
% mu: vector of prior mean values (nx1)
% OUTPUT:
% k: function handle for prior kernel loaded with grid-search-optimized parameters
function k = compute_kernel_parameters(x,y,mu)
s_ytest=0.2:.1:1;
s_ntest=0; %0.2:.1:1;
ltest=0.1:.1:1.2;

maxl=-Inf;
maxsn=NaN;maxsy=maxsn;maxlen=maxsy;

n=numel(x);

likelihood=NaN(numel(ltest),numel(s_ntest),numel(s_ytest));

for i=1:numel(s_ytest)
    for j=1:numel(s_ntest)
        for l=1:numel(ltest)
            k = @(x_1, x_2) kernel(s_ytest(i), ltest(l), s_ntest(j), x_1, x_2); % kernel params to test
            sigma = bsxfun(k, x, x');
            likelihood(l,j,i)=(-1/2)*log(det(sigma))-((1/2)*(y-mu).')*(sigma\(y-mu))-(n/2)*log(2*pi);
            if (likelihood(l,j,i) > maxl)
                maxsy=s_ytest(i);
                maxsn=s_ntest(j);
                maxlen=ltest(l);
                maxl=likelihood(l,j,i);
            end
        end
    end
end
%k = @(x_1, x_2) kernel(maxsy, maxlen, maxsn, x_1, x_2); % kernel function with optimal parameters
k = @(x_1, x_2) kernel(maxsy, maxlen, maxsn, x_1, x_2);
end