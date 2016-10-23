function s = sample_h(v, W, h_bias)
    p = cond_p(W, v, h_bias);
    s = rand(length(p), 1);
    s = (s < p) * 1;
end