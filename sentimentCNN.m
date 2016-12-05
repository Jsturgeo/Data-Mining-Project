function res = sentimentCNN(X, convW, convB, outW, outB, dims, label)


poolRes = cell(1, dims.numFilterSizes);
reluRes = cell(1, dims.numFilterSizes);
convRes = cell(1, dims.numFilterSizes);
    
%% section 2.1 forward propagation and compute the loss
for j=1:dims.numFilterSizes
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
    
% max pooling to create feature vector
poolResCombined = vl_nnconcat(poolRes, 3);   
featureVec = reshape(poolResCombined, [dims.totalFilters, 1]); 

% Apply weights to feature vector to get final output
output = transpose(outW)*featureVec + outB;
res.output = reshape(output, [1, 1, numel(output)]);

%% Backpropagation
    % loss from the output layer
if nargin > 6
   
    dEdo = vl_nnsoftmaxloss(res.output, label,1);
    res.dEdo = reshape(dEdo, [dims.numClasses, 1]);
    % backward through the weights before output layer
    dEdz = outW * res.dEdo;
    dEdz = reshape(dEdz, [1, 1, numel(dEdz)]);
    
    res.dEdw = featureVec*transpose(res.dEdo);
    
    % Backwards through concatenation of 1-max pooling
    dEdp = vl_nnconcat(poolRes, 3, dEdz);
    
     for j=1:dims.numFilterSizes
        
         % Backward through 1-max pool on feature maps
         convSize = size(convRes{j});
         dzdpool = vl_nnpool(reluRes{j}, [convSize(1), 1], dEdp{j});
     
         % Backward through activation function
         dzdrelu = vl_nnrelu(convRes{j}, dzdpool);
         
         % Backward through convolution
         [res.dzdx{j}, res.dzdw{j}, res.dzdb{j}] = vl_nnconv(X, convW{j}, convB{j}, dzdrelu);
     end

end
