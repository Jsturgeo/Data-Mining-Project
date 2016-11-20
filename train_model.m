%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

%clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

%% TODO: split data to perform cross-validation
[data, wordMap] = read_data('train.txt');
padVal = '#pad#';

% Create initial word embedding
T = customWordvecs;
d = 300;
numWords = length(wordMap);
% random sample from normal distribution 
% with mean=0, variance=0.1
%T = normrnd(0, 0.1, [numWords, d]);

% Initialize filters
filterSizes = [2, 3, 4];
numFilterSizes = length(filterSizes);
numFilters = 2;

convW = cell(length(filterSizes), 1);
convB = cell(length(filterSizes), 1);

% Learning parameters
eta = 0.0001;

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

%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

%% for each example in train.txt do
for ind=1:length(data)
    [i, sentence, label] = data{ind,:};
    sentence;
    label = label + 1;
    sentenceLength = length(sentence);
    if sentenceLength < max(filterSizes)
        numPad = max(filterSizes) - sentenceLength;
        padCell = cell(1, numPad);
        [padCell{1:numPad}] = deal(padVal);
        sentence = [sentence padCell];
        data{ind, 2} = sentence;
        sentenceLength = length(sentence);
    end
    sentenceWordInds = zeros(sentenceLength, 1);
    % get index for each word in the sentence
    for w=1:sentenceLength
        sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
    end
    X = T(sentenceWordInds, :);
    
    poolRes = cell(1, numFilterSizes);
    reluRes = cell(1, numFilterSizes);
    convRes = cell(1, numFilterSizes);
    
    %% section 2.1 forward propagation and compute the loss
    for j=1:numFilterSizes
        % convolutional operation
        conv = vl_nnconv(X, convW{j}, convB{j});        
        % apply activation function: relu
        relu = vl_nnrelu(conv);       
        % 1-max pooling operation
        convSize = size(conv);
        pool = vl_nnpool(relu, [convSize(1), 1]);      
        % important: keep these values for back-prop
        reluRes{j} = relu;
        convRes{j} = conv;
        poolRes{j} = pool;  
    end
    
    poolResCombined = vl_nnconcat(poolRes, 3);   
    featureVec = reshape(poolResCombined, [totalFilters, 1]); 
    output = transpose(outW)*featureVec + outB;
    output = reshape(output, [1, 1, numel(output)]);
    loss = vl_nnsoftmaxloss(output, label); 
    
    %% section 2.2 backward propagation and com
     
    % Compute the derivatives
    
    % Backward through softmax on single feature vector
    
    dEdo = vl_nnsoftmaxloss(output, label, 1);
    dEdo = reshape(dEdo, [numClasses, 1]);
    
    dEdz = outW * dEdo;
    dEdz = reshape(dEdz, [1, 1, numel(dEdz)]);
    
    dEdw = featureVec*transpose(dEdo);
    
    dEdp = vl_nnconcat(poolRes, 3, dEdz);
    
    for j=1:numFilterSizes
        % Backward through 1-max pool on feature maps
        convSize = size(convRes{j});
        dzdpool = vl_nnpool(reluRes{j}, [convSize(1), 1], dEdp{j});
    
        % Backward through activation function
        dzdrelu = vl_nnrelu(convRes{j}, dzdpool);
        
        % Backward through convolution
        [dzdx, dzdw, dzdb] = vl_nnconv(X, convW{j}, convB{j}, dzdrelu);

        %% section 2.3 update the parameters
        
        outW = outW - eta * dEdw;
        outB = outB - eta * dEdo;
        convW{j} = convW{j} - eta * dzdw;
        convB{j} = convB{j} - eta * dzdb;
        X = X - eta * dzdx;
        
        %for w=1:sentenceLength
        %    T(sentenceWordInds(w), :) = X(w,:);
        %end
        
    end
end %end for

save('custom_embedding.mat', 'T', 'wordMap');
save('weights.mat', 'convW', 'convB', 'outW', 'outB');
    
