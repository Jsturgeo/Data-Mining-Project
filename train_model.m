%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code
% Adagrad implemented from this pseudocode: https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
% Other possible avenues to consider (from the original paper):
%   - using more filters (closer to 100 instead of just 2)
%   - experiment with finetuning the word2vec vectors
%   - add in glove vectors as separate channel
%   - make sure randomly initialized vectors have the same variance as the
%   pre-trained ones
%   - add 50% dropout to avoid overfitting


%clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

%[data, wordMap] = read_data('train.txt', 1);

% Split data into k folds
kfolds=5;
numSamples = size(data,1);
length_fold = numSamples / kfolds; 
data = data(randperm(numSamples),:);
s =0;
accuracy = zeros(kfolds,1);
    



    % String for padding sentences that are too short
    padVal = '#pad#';

    % Create initial word embedding
    load('customWord2Vec.mat');
    T = customWordvecs;
    d = 300;
    numWords = length(wordMap);

    % Initialize filters
    dims.filterSizes = [2, 3, 4];
    dims.numFilterSizes = length(dims.filterSizes);
    dims.numFilters = 2;

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
    outB = zeros(dims.numClasses, 1);

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

    %% Section 2: training
    % Note: 
    % you may need the resouces [2-4] in the project description.
    % you may need the follow MatConvNet functions: 
    %       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

    % TODO: find best number of epochs to perform
 for k= 1:kfolds
    testData = data(s+1:length_fold*k,:);
    if (k>1)
        trainData = data([1:s length_fold*k+1:end],:);
    else
        trainData = data(length_fold+1:end,:);
    end
    s =+ length_fold;   
    for epoch=1:100
        %shuffle for each epoch
        trainData = trainData(randperm(size(trainData,1)),:);
    
        for ind=1:length(trainData)
            [i, sentence, label] = trainData{ind,:};
            label = label + 1;  %add 1 because matlab uses indexing at 1
            sentenceLength = length(sentence);
            % Pad sentence if sentence is too short for filters
            % TODO: put in separate function?
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

            %% section 2.3 update the parameters
            
            % TODO: should we be updating T (word embedding) as well?
            for j=1:dims.numFilterSizes 
                historical_grad.dzdw{j} = historical_grad.dzdw{j} + (res.dzdw{j}).^2;
                convW{j} = convW{j} - eta * (res.dzdw{j} ./ (fudge_factor + sqrt(historical_grad.dzdw{j})));
                historical_grad.dzdb{j} = historical_grad.dzdb{j} + (res.dzdb{j}).^2;
                convB{j} = convB{j} - eta * (res.dzdb{j} ./ (fudge_factor + sqrt(historical_grad.dzdb{j})));  
                X = X - eta * res.dzdx{j};
            end
            for w=1:sentenceLength
                T(sentenceWordInds(w), :) = X(w,:);
            end
            
            historical_grad.dEdw = historical_grad.dEdw + (res.dEdw).^2;
            outW = outW - eta * (res.dEdw ./ (fudge_factor + sqrt(historical_grad.dEdw)));
            historical_grad.dEdo = historical_grad.dEdo + (res.dEdo).^2;
            outB  = outB - eta * (res.dEdo ./ (fudge_factor + sqrt(historical_grad.dEdo)));

        end 

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
        % TODO: put in separate function?
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
        % TODO: should this abs() be here?
        [~, output] = max(res.output);
        predLabels(ind) = output - 1;
    end 

    accuracy(k) = sum(predLabels == trueLabels)/length(trueLabels);
    
end
%% Once we have trained good parameters, save them.
%save('custom_embedding.mat', 'T', 'wordMap');
%save('weights.mat', 'convW', 'convB', 'outW', 'outB', 'dims', 'padVal');
