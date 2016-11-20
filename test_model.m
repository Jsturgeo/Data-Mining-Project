% clear; clc;
load('customWord2Vec.mat');
load('weights.mat');

filterSizes = [2, 3, 4];
numFilterSizes = length(filterSizes);
numFilters = 2;

padVal = '#pad#';

%[data, ~] = read_data('train.txt');
T = customWordvecs;
labels = zeros(length(data),1);
[lengthT, dimT] = size(T);

for ind=1:length(data)
    [i, sentence, trueLabel] = data{ind,:};
    sentence
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
    
    poolRes = cell(1, numFilterSizes);
    %% section 2.1 forward propagation and compute the loss
    for j=1:numFilterSizes
        % Skip the filter if it's too big
        if filterSizes(j) > sentenceLength
            poolRes{j} = 0; 
           continue 
        end
        % convolutional operation
        conv = vl_nnconv(X, convW{j}, convB{j});
        
        % apply activation function: relu
        relu = vl_nnrelu(conv);
        
        % 1-max pooling operation
        convSize = size(conv);
        pool = vl_nnpool(relu, [convSize(1), 1]);
        
        % important: keep these values for back-prop
        poolRes{j} = pool;  
    end
    
    poolResCombined = vl_nnconcat(poolRes, 3);   
    featureVec = reshape(poolResCombined, [totalFilters, 1]); 
    output = transpose(outW)*featureVec + outB;
    [~, output] = max(abs(output));
    
    labels(ind) = output - 1;
end