function [data, wordMap] = read_data(fileName, trainData)
% Based on code provided by Jiaxi Tang.
% Reads data from file with fileName and returns parsed data
% and map of vocabulary (if applicalble)
% if trainData flag is specified, parsed data includes labels
% and the wordmap is populated
% Otherwise, parsed data does not include labels and vocabulary
% is not compiled
% return:
%    If trainData is specified: 
%       data(cell), 1st column -> sentence id, 
                    2nd column -> words, 
                    3rd column -> label
%       wordMap(Map), contains all words and their index, 
                      get word index by calling wordMap(word)
%    Otherwise:
%       data(cell), 1st column -> sentence id, 
                    2nd column -> words
%       wordMap(NaN), dummy variable

headLine = true;
separater = '::';
padVal = '#pad#';
maxCells = 6000;

words = [];
if trainData
    data = cell(maxCells, 3);
else
    data = cell(maxCells, 2);
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
% Remove empty data cells
emptyCells = cellfun('isempty', data);
data(all(emptyCells,2),:) = [];

% Compile word map (if applicable)
if trainData
    words = unique(words);
    wordMap = containers.Map(words, 1:length(words));
    wordMap(padVal) = length(wordMap)+1;
else
    wordMap = NaN;
end

fprintf('Finished loading data and vocabulary\n');