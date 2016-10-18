tmp = csvread('data/digitstrain.txt');
tmp = (tmp > 0.5) * 1;
train = struct;
train.X = tmp(:, 1:end-1);
train.y = tmp(:, end);

tmp = csvread('data/digitsvalid.txt');
tmp = (tmp > 0.5) * 1;
valid = struct;
valid.X = tmp(:, 1:end-1);
valid.y = tmp(:, end);

params = struct;
params.h_num = 100;
params.step = 1e-2;
params.max_iter = 500;
params.k = 5;

model = rbm_learn(train, valid, params);
display_network(W);