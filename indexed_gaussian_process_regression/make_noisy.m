function output = make_noisy(mapping, s_n, x)

output = mapping(find(ismember(mapping, x)), 2) ...
    + normrnd(0, s_n, length(x), 1);



