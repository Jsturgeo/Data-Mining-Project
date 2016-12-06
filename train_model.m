% This script trains a convolutional neural network and saves
% the weights
% author: Anne-Marie Therien-Daniel and Julie Sturgeon
% date: December 6, 2016


clear; clc;

% Get training data
[data, wordMap] = read_data('train.txt', 1);
dataLength = length(data);

% String for padding sentences that are too short
padVal = '#pad#';

% Load initial word2vec word embedding
load('customWord2Vec.mat');
T = customWordvecs;
d = 300;
numWords = length(wordMap);

% Initialize filters
dims.filterSizes = [2, 3, 4, 5];
dims.numFilterSizes = length(dims.filterSizes);
dims.numFilters = 50;

convW = cell(length(dims.filterSizes), 1);
convB = cell(length(dims.filterSizes), 1);

for i = 1:length(dims.filterSizes)
    filterSize = dims.filterSizes(i);
    % initialize W with: FW x FH x FC x K
    convW{i} = normrnd(0, 0.1, [filterSize, d, 1, dims.numFilters]);
    % initialize bias B as K x 1
    convB{i} = zeros(dims.numFilters, 1);
end

% Initialize output layer
dims.totalFilters = length(dims.filterSizes) * dims.numFilters;
dims.numClasses = 2;
outW = normrnd(0, 0.1, [dims.totalFilters, dims.numClasses]);
outB = ones(dims.numClasses, 1);

% Initial parameters needed for Adagrad
% Initial learning parameter eta for Adagrad
eta = 0.01; % initial setting, can probably stay as-is
fudge_factor = 1e-6; % for numerical stability
% Initialize historical gradients to be zero for each of the parameters
% that we need to update
historical_grad.dzdw = cell(length(dims.filterSizes), 1);
historical_grad.dzdb = cell(length(dims.filterSizes), 1);
for i = 1:length(dims.filterSizes)
    filterSize = dims.filterSizes(i);
    % initialize W with: FW x FH x FC x K
    historical_grad.dzdw{i} = zeros(filterSize, d, 1, dims.numFilters);
    % initialize bias B as K x 1
    historical_grad.dzdb{i} = zeros(dims.numFilters, 1);
end
historical_grad.dEdw = zeros(dims.totalFilters, dims.numClasses);
historical_grad.dEdo = zeros(dims.numClasses, 1);   

numEpochs = 50;
accuracy = zeros(numEpochs,1);

% Split data into k folds for cross-validation
kfolds=5;
numSamples = size(data,1);
length_fold = numSamples / kfolds; 

s = 0;
accuracy = zeros(kfolds,numEpochs);
    
for k= 1:kfolds 
    testData = data(s+1:length_fold*k,:);
    if (k>1)
        trainData = data([1:s length_fold*k+1:end],:);
    else
        trainData = data(length_fold+1:end,:);
    end
    s =+ length_fold;

    %% Training

    for epoch=1:numEpochs
        %shuffle for each epoch
        trainData = trainData(randperm(size(trainData,1)),:);
    
        for ind=1:length(trainData)
            [i, sentence, label] = trainData{ind,:};
            label = label + 1;  %add 1 because matlab starts indexing at 1
            sentenceLength = length(sentence);
            % Pad sentence if sentence is too short for filters
            if sentenceLength < max(dims.filterSizes)
                numPad = max(dims.filterSizes) - sentenceLength;
                padCell = cell(1, numPad);
                [padCell{1:numPad}] = deal(padVal);
                sentence = [sentence padCell];
                trainData{ind, 2} = sentence;
                sentenceLength = length(sentence);
            end
            sentenceWordInds = zeros(sentenceLength, 1);
            % get index for each word in the sentence
            for w=1:sentenceLength
                sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
            end 
            X = T(sentenceWordInds, :);
            % Run sample through CNN (forward and backward)
            res = sentimentCNN(X, convW, convB, outW, outB, dims, label);

            %% Update the parameters
            % For each parameter to update, we add the current gradient to the historical gradient
            % first, then update the current value
            % update convolutional filters
            for j=1:dims.numFilterSizes 
                historical_grad.dzdw{j} = historical_grad.dzdw{j} + (res.dzdw{j}).^2;
                convW{j} = convW{j} - eta * (res.dzdw{j} ./ (fudge_factor + sqrt(historical_grad.dzdw{j})));
                historical_grad.dzdb{j} = historical_grad.dzdb{j} + (res.dzdb{j}).^2;
                convB{j} = convB{j} - eta * (res.dzdb{j} ./ (fudge_factor + sqrt(historical_grad.dzdb{j})));  
                % Update word embedding using SGD from each convolutional filter
                X = X - eta * res.dzdx{j};
            end
            % Map updated word vectors back to master word mapping
            for w=1:sentenceLength
                T(sentenceWordInds(w), :) = X(w,:);
            end
            % Update output layer 
            historical_grad.dEdw = historical_grad.dEdw + (res.dEdw).^2;
            outW = outW - eta * (res.dEdw ./ (fudge_factor + sqrt(historical_grad.dEdw)));
            historical_grad.dEdo = historical_grad.dEdo + (res.dEdo).^2;
            outB  = outB - eta * (res.dEdo ./ (fudge_factor + sqrt(historical_grad.dEdo)));

        end 

        %% VALIDATION OF MODEL
        % store labels to compute accuracy later
        trueLabels = zeros(length(testData), 1);
        predLabels = zeros(length(testData), 1);
    
        for ind=1:length(testData)
            [i, sentence, label] = testData{ind,:};
            trueLabels(ind) = label;
            sentenceLength = length(sentence);
            % Pad sentence if sentence is too short for filters
            if sentenceLength < max(dims.filterSizes)
                numPad = max(dims.filterSizes) - sentenceLength;
                padCell = cell(1, numPad);
                [padCell{1:numPad}] = deal(padVal);
                sentence = [sentence padCell];
                testData{ind, 2} = sentence;
                sentenceLength = length(sentence);
            end
            sentenceWordInds = zeros(sentenceLength, 1);
            % get index for each word in the sentence
            for w=1:sentenceLength
                sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
            end
            X = T(sentenceWordInds, :);
            %Run sample through CNN (forward only)
            res = sentimentCNN(X, convW, convB, outW, outB, dims);

            % Get index of greatest value from out 2x1 output vector
            [~, output] = max(res.output);
            %Subtract 1 to get label in {0,1} instead of {1,2}
            predLabels(ind) = output - 1; 
        end
        accuracy(k, epoch) = sum(trueLabels == predLabels)/length(testData)
    end
end
    
%% Once we have trained good parameters, save them.
save('custom_embedding.mat', 'T', 'wordMap');
save('weights.mat', 'convW', 'convB', 'outW', 'outB', 'dims', 'padVal');
