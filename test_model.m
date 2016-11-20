% clear; clc;
load('custom_embedding.mat');
load('weights.mat');

filterSizes = [2, 3, 4];
numFilterSizes = length(filterSizes);
numFilters = 2;

padVal = '#pad#';

%[data, ~] = read_data('train.txt');
data = testData;
predLabels = zeros(length(data),1);
trueLabels = zeros(length(data),1);
[lengthT, dimT] = size(T);

for ind=1:length(data)
    [i, sentence, trueLabel] = data{ind,:};
    sentence
    trueLabels(i) = trueLabel;
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
        if isKey(wordMap, strjoin(sentence(w)))
            sentenceWordInds(w) = wordMap(strjoin(sentence(w)));
        else
            % If needed word is not in 
            newInd = length(wordMap) + 1;
            sentenceWordInds(w) = newInd;
            T(newInd,:) = normrnd(0, 0.1, [1, dimT]);
            wordMap(strjoin(sentence(w))) = newInd;
        end
    end
    X = T(sentenceWordInds, :);
    
    res = sentimentCNN(X, convW, convB, outW, outB);
    
    [~, output] = max(abs(res.output));
    
    predLabels(ind) = output - 1;
end