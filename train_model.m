%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

[data, wordMap] = read_data('train.txt');

% Split data into training and testing sets
numSamples = size(data,1);
numTrain = 0.9 * numSamples;

data = data(randperm(numSamples),:);
trainData = data(1:numTrain,:);
testData = data(numTrain + 1:end,:);

% String for padding sentences that are too short
padVal = '#pad#';

% Create initial word embedding
load('customWord2Vec.mat');
T = customWordvecs;
d = 300;
numWords = length(wordMap);

% Initialize filters
filterSizes = [2, 3, 4];
numFilterSizes = length(filterSizes);
numFilters = 2;

convW = cell(length(filterSizes), 1);
convB = cell(length(filterSizes), 1);

for i = 1:length(filterSizes)
    filterSize = filterSizes(i);
    % initialize W with: FW x FH x FC x K
    convW{i} = normrnd(0, 0.1, [filterSize, d, 1, numFilters]);
    % initialize bias B as K x 1
    convB{i} = zeros(numFilters, 1);
end

% Initialize output layer
totalFilters = length(filterSizes) * numFilters;
numClasses = 2;
outW = normrnd(0, 0.1, [totalFilters, numClasses]);
outB = zeros(numClasses, 1);

% Learning parameters
% TODO: find best value of eta
eta = 0.0001;

%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

% TODO: find best number of epochs to perform
for epoch=1:1
    for ind=1:length(trainData)
        [i, sentence, label] = trainData{ind,:};
        label = label + 1;
        sentenceLength = length(sentence);
        % Pad sentence if sentence is too short for filters
        % TODO: put in separate function?
        if sentenceLength < max(filterSizes)
            numPad = max(filterSizes) - sentenceLength;
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
        res = sentimentCNN(X, convW, convB, outW, outB, label);

        %% section 2.3 update the parameters
        % TODO: should we be updating T (word embedding) as well?
        for j=1:numFilterSizes     
            convW{j} = convW{j} - eta * res.dzdw{j};
            convB{j} = convB{j} - eta * res.dzdb{j};  
        end
        outW = outW - eta * res.dEdw;
        outB = outB - eta * res.dEdo;
        
    end 
end

%% TESTING
% store labels to compute accuracy later
trueLabels = zeros(length(testData), 1);
predLabels = zeros(length(testData), 1);

for ind=1:length(testData)
    [i, sentence, label] = testData{ind,:};
    trueLabels(ind) = label;
    sentenceLength = length(sentence);
    % Pad sentence if sentence is too short for filters
    % TODO: put in separate function?
    if sentenceLength < max(filterSizes)
        numPad = max(filterSizes) - sentenceLength;
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
    res = sentimentCNN(X, convW, convB, outW, outB);
    
    % Get index of greatest value from out 2x1 output vector
    % TODO: should this abs() be here?
    [~, output] = max(abs(res.output));
    predLabels(ind) = output - 1;
end 

accuracy = sum(predLabels == trueLabels)/length(trueLabels);

%% Once we have trained good parameters, save them.
%save('custom_embedding.mat', 'T', 'wordMap');
%save('weights.mat', 'convW', 'convB', 'outW', 'outB');
    
