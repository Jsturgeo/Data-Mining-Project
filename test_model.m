clear; clc;

inFileName = 'sample_test.txt';
outFileName = 'out1.txt';

load('custom_embedding.mat');
load('weights.mat');

[data, ~] = read_data(inFileName, 0);
[lengthT, dimT] = size(T);

predictions = zeros(2, length(data));

for ind=1:length(data)
    [i, sentence] = data{ind,:};
    predictions(1, ind) = i;
    sentenceLength = length(sentence);
    if sentenceLength < max(dims.filterSizes)
        numPad = max(dims.filterSizes) - sentenceLength;
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
    
    res = sentimentCNN(X, convW, convB, outW, outB, dims);
    
    [~, output] = max(res.output);
    
    predictions(2, ind) = output - 1;
end

outFile = fopen(outFileName, 'w');

formatSpec = '%d::%d\n';
fprintf(outFile,formatSpec, predictions);
