function plothelp(W)
[p, n] = size(W);
for i = 1:10
    for j = 1:10
        idx = (i - 1) * 10 + j;
        subplot(10, 10, (i - 1) * 10 + j);
        imshow(reshape(W(:, idx), 28, 28), [min(W(:,idx)), max(W(:,idx))]);
    end
end
end