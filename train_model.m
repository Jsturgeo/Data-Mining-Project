%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

[data, wordMap] = read_data;
numWords = length(wordMap);


%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

for ind=1:length(data)
    [i, sentence, label] = data{ind,:};
    
    %% Encode a one-hot vector for
    sentenceLength = length(sentence);
    sentenceWordInds = zeros(sentenceLength, 1);
    %Get index of each word in the sentence
    for w=1:sentenceLength
        sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
    end
    % Build one-hot matrix where each row is a one-hot vector for each word
    ohVec = sparse(1:sentenceLength, sentenceWordInds, ones(sentenceLength, 1), sentenceLength, numWords);
    
    %% for each example in train.txt do
    %% section 2.1 forward propagation and compute the loss
    % Create a bank of 2 <numWords> x 3 filters (to start with)
    w = randn(numWords,2,1,2,'single');
    
    % Apply the convolution operator
    y = vl_nnconv(ohVec, w, []) ; %bias terms?

    %% section 2.2 backward propagation and compute the derivatives
    % TODO: your code

    %% section 2.3 update the parameters
    % TODO: your code
end %end for