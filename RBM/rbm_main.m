%% Load data
tmp = csvread('../data/digitstrain.txt');
train = struct;
train.X = tmp(:, 1:end-1);
train.y = tmp(:, end);
train.X = (train.X > 0.5) * 1;

tmp = csvread('../data/digitsvalid.txt');
valid = struct;
valid.X = tmp(:, 1:end-1);
valid.y = tmp(:, end);
valid.X = (valid.X > 0.5) * 1;

params = struct;
params.h_num = 100;
params.step = 1e-2;
params.max_iter = 50;
params.k = 5;

hls = [50, 100, 200, 500];
models = cell(1,4);
for idx = 1:4
    params.h_num = hls(idx);
    models{idx} = rbm_learn(train, valid, params);
end
save('models.mat', 'models');
% display_network(model.W);

%% Plotting error
plot(model.err_t)
hold on
plot(model.err_v)
legend('training error', 'validation error')
xlabel('epochs')
ylabel('cross-entropy')
title('rbm CD-5 Hidden 100')

%% Sampling
imgs = zeros(784, 100);
for i = 1:100
    v = randi([0,1], 784, 1);
%     rd = randi([0,3000],1,1);
%     v = digits(:,rd);
    for j = 1:1000
%         h = cond_p(model.W, v, model.h_bias);
%         v = cond_p(model.W', h, model.v_bias);
        h = sample_h(v, model.W, model.h_bias);
        v = sample_v(h, model.W, model.v_bias);
    end
%     h = sample_h(v, model.W, model.h_bias);
%     v = cond_p(model.W', h, model.v_bias);
%     v = (v > 0.5);
    imgs(:, i) = v;
end
display_network(imgs, false, true);
