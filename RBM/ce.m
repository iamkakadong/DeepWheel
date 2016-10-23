function l = ce(v, W, h, b)
    p = cond_p(W', h, b);
    l = v' * log(p) + (1 - v)' * log(1 - p);
    l = -l;
end