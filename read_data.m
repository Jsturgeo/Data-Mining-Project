function [data, wordMap] = read_data(fileName, trainData)
% CMPT-741 example code for: reading training data and building vocabulary.
% NOTE: reading testing data is similar, but no need to build the vocabulary.
%
% return: 
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       wordMap(Map), contains all words and their index, get word index by calling wordMap(word)

headLine = true;
separater = '::';
padVal = '#pad#';

words = [];
if trainData
    data = cell(6000, 3);
else
    data = cell(6000, 2);
end

fid = fopen(fileName, 'r');
line = fgets(fid);

ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);
    sid = str2double(attrs{1});
    
    s = attrs{2};
    w = strsplit(s);
    words = [words w];
    
    if length(attrs) > 2
        y = str2double(attrs{3});
    else
        y = NaN;
    end
    
    % save data
    data{ind, 1} = sid;
    data{ind, 2} = w;
    if trainData
        data{ind, 3} = y;
    end
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
emptyCells = cellfun('isempty', data);
data(all(emptyCells,2),:) = [];
if trainData
    words = unique(words);
    wordMap = containers.Map(words, 1:length(words));
    wordMap(padVal) = length(wordMap)+1;
else
    wordMap = NaN;
end

fprintf('finish loading data and vocabulary\n');