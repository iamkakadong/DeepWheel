function s = sample_v(h, W, v_bias)
    p = cond_p(W', h, v_bias);
    s = rand(length(p), 1);
    s = (s < p) * 1;
end