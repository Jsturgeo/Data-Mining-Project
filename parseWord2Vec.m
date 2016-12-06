function [customWordvecs, customWordvecsnorm, newWords] = parseWord2Vec(wordMap, word2Index, wordvecs, wordvecsnorm)
allWords = keys(wordMap);
wvDim = 300;
wvstd = mean(std(wordvecs));
wvnstd = mean(std(wordvecsnorm));
customWordvecs = zeros(length(allWords), wvDim);
customWordvecsnorm = zeros(length(allWords), wvDim);
newWords = [];
for w=1:length(allWords)
    word = char(allWords(w));
    i = wordMap(word);
    if isKey(word2Index, word)
        ind = word2Index(word);
        customWordvecs(i,:) = wordvecs(ind,:);
        customWordvecsnorm(i,:) = wordvecsnorm(ind,:);
    else
        newWords = [newWords word];
        customWordvecs(i,:) = normrnd(0, wvstd, [1, wvDim]);
        customWordvecsnorm(i,:) = normrnd(0, wvnstd, [1, wvDim]);
    end
end
