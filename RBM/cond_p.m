function p = cond_p(W, layer, bias)  
    m = W' * layer + bias;
    p = sigmoid(m);
end
