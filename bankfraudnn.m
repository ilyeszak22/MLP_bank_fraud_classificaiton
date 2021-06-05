%% Machine Learning NN bank fraud detection 

t = cputime;
%% Setup the parameters you will use for this exercise
input_layer_size  = 29;  % 29 inputs 
hidden_layer_size = 10;   % 50 hidden units
num_labels = 1;  % binary output
iteration= 100;




%% ================  Initializing Pameters ================

% creating random initialisaiton of Theta1/2 
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================: Training NN ===================

%
fprintf('\nTraining Neural Network... \n')

%  choose a relative small "interation" <100 for fast computing 

options = optimset('MaxIter', iteration);

%  try differents vallues of lambda for better accuracy 0<lambda<20
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
             
accuracy=testaccuracy(X,y,Theta1,Theta2)
accuracytestset=testaccuracy(Xtest,ytest,Theta1,Theta2)
e = cputime-t;


fprintf('time processing : %f min \n \n', e/60);



