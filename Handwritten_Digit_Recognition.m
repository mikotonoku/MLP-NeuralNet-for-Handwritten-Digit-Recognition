% Data loading
data = load('datapiscisla_all.mat');
max_pre_pixel = 255;                                                        % Maximum pixel value in the image before normalization

% Data normalization
data.XDataall = data.XDataall / max_pre_pixel;                              % Normalize to range <0;1>

% Create and configure MLP neural network
hidden_neurons = 65;                                                        % Number of neurons in the hidden layer
net = patternnet(hidden_neurons);                                           % Create MLP network

% Set data division using 'dividerand'
net.divideFcn = 'dividerand';                                               % Randomly split the dataset
net.divideParam.trainRatio = 0.6;                                           % 60% of data for training
net.divideParam.valRatio = 0;                                               % 0% for validation
net.divideParam.testRatio = 0.4;                                            % 40% of data for testing

% Set training parameters
net.trainParam.goal = 0.00001;                                              % Desired error goal
net.trainParam.epochs = 1000;                                               % Maximum number of training epochs
net.trainParam.max_fail = 12;                                               % Maximum validation failures

% Train the neural network
[net, tr] = train(net, data.XDataall, data.YDataall);

% Display the network structure
view(net);

% Compute accuracy on training data
train_outputs = net(data.XDataall(:, tr.trainInd));                             
train_accuracies = 1 - confusion(data.YDataall(:, tr.trainInd), train_outputs);
train_accuracies = train_accuracies * 100;

% Compute accuracy on testing data
test_outputs = net(data.XDataall(:, tr.testInd));                               
test_accuracies = 1 - confusion(data.YDataall(:, tr.testInd), test_outputs);
test_accuracies = test_accuracies * 100;
testError = 100 - test_accuracies;                                              

% Display confusion matrix for test data
figure;
plotconfusion(data.YDataall(:, tr.testInd), test_outputs);
title(sprintf('Test Data\nError: %.2f%%', testError));

% Display min, mean, and max accuracies
fprintf('Training Accuracy: min = %.2f%%, mean = %.2f%%, max = %.2f%%\n', ...
    min(train_accuracies), mean(train_accuracies), max(train_accuracies));
fprintf('Testing Accuracy: min = %.2f%%, mean = %.2f%%, max = %.2f%%\n', ...
    min(test_accuracies), mean(test_accuracies), max(test_accuracies));

%--------------------------------------------------------------------------

unique_labels = 0:9; % Digits from 0 to 9
samples_per_label = []; % Store samples per digit class

fprintf('\nTesting at least one sample from each digit:\n');
for label = unique_labels
    % Pick one sample from each digit
    idx = find(data.YDataall(label+1, :) == 1, 1);                          % First occurrence of the digit
    sample = data.XDataall(:, idx);                                         % Input sample
    actual_label = label;                                                   % Actual class
    predicted_output = net(sample);                                         % Network prediction
    [~, predicted_label] = max(predicted_output);                           % Predicted class
    predicted_label = predicted_label - 1;                                  % Adjust to match label indexing

    % Store results
    samples_per_label = [samples_per_label; actual_label, predicted_label];

    % Display results
    fprintf('Digit: %d, Predicted: %d\n', actual_label, predicted_label);
end
