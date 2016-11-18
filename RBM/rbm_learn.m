function [ model ] = rbm_learn( train, valid, param )
%RBM Summary of this function goes here
%   Detailed explanation goes here

h_num = param.h_num;
step = param.step;
max_iter = param.max_iter;
[n, p] = size(train.X);
[vn, ~] = size(valid.X);

W = (rand(p, h_num) - 0.5) * sqrt(6) / sqrt(p + h_num);    % Initialize weight matrix p * h
v_bias = zeros(p, 1);    % p * 1
h_bias = zeros(h_num, 1);    % h * 1

err_t = zeros(max_iter, 1);
err_v = zeros(max_iter, 1);

for iter = 1 : max_iter
    for i = randperm(n)
        v0 = train.X(i, :)'; % p * 1
        v = v0;
        for k = 1 : param.k
            h = sample_h(v, W, h_bias);  % h * 1
            v = sample_v(h, W, v_bias);  % p * 1
        end
        p0 = cond_p(W, v0, h_bias);
        pk = cond_p(W, v, h_bias);
        W = W + step * (p0 * v0' - pk * v')';
        h_bias = h_bias + step * (p0 - pk);
        v_bias = v_bias + step * (v0 - v);
    end
    
    err = 0;
    for i = 1 : n
        vt = train.X(i, :)';
        h = sample_h(vt, W, h_bias);
        err = err + ce(vt, W, h, v_bias);
    end
    err_t(iter) = err / n;

    err = 0;
    for i = 1 : vn
        vv = valid.X(i, :)';
        h = sample_h(vv, W, h_bias);
        err = err + ce(vv, W, h, v_bias);
    end
    err_v(iter) = err / vn;
    fprintf('current iteration %d, train error %0.5f, valid error %0.5f \n', iter, err_t(iter), err_v(iter));
end

model = struct;
model.W = W;
model.v_bias = v_bias;
model.h_bias = h_bias;
model.err_t = err_t;
model.err_v = err_v;

end