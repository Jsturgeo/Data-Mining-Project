%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

%clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

[data, wordMap] = read_data;


% Create initial word embedding
d = 5;
numWords = length(wordMap);
% random sample from normal distribution 
% with mean=0, variance=0.1
T = normrnd(0, 0.1, [numWords, d]);

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

for ind=1:length(data)
    [i, sentence, label] = data{ind,:}
    sentenceLength = length(sentence);
    sentenceWordInds = zeros(sentenceLength, 1);
    % get index for each word in the sentence
    for w=1:sentenceLength
        sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
    end
    X = T(sentenceWordInds, :);
    
    poolRes = cell(1, numFilterSizes);
    cache = cell(2, numFilterSizes);
    
    for j=1:numFilterSizes
        % convolutional operation
        if filterSizes(j) > sentenceLength
            cache{2, j} = 0;
            cache{1, j} = 0;
            poolRes{j} = 0; 
           continue 
        end
        conv = vl_nnconv(X, convW{j}, convB{j});
        
        % apply activation function: relu
        relu = vl_nnrelu(conv);
        
        % 1-max pooling operation
        convSize = size(conv);
        pool = vl_nnpool(relu, [convSize(1), 1]);
        
        % important: keep these values for back-prop
        cache{2, j} = relu;
        cache{1, j} = conv;
        poolRes{j} = pool;  
    end
    

    
    %% for each example in train.txt do
    %% section 2.1 forward propagation and compute the loss


    %% section 2.2 backward propagation and compute the derivatives
    % TODO: your code

    %% section 2.3 update the parameters
    % TODO: your code
end %end for
    
