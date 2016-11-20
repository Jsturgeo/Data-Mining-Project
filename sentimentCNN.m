function res = sentimentCNN(x, convW, convB, outW, outB, dEdo)
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

if nargin > 5
end
end