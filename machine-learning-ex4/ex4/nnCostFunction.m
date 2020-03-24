function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
k=0;
X=[ones(m,1) X];
theta1=Theta1(:,2:input_layer_size + 1);
theta1=theta1.^2;
theta2=Theta2(:,2:hidden_layer_size + 1);
theta2=theta2.^2;
Delta2=zeros(size(Theta2));
Delta1=zeros(size(Theta1));
for i=1:m,
  Y=zeros(num_labels,1);
  x=X(i,:);
  Y(y(i))=1;
  Z=Theta1*x';
  Z=[1:Z];
  A=sigmoid(Theta1*x');
  A=[1;A];
  H=sigmoid(Theta2*A);
  K=Y.*log(H)+(1-Y).*log(1-H);
  k=k+sum(K);
  delta3=H-Y;
  delta2=(Theta2(:,2:hidden_layer_size + 1))'*delta3.*sigmoidGradient(Theta1*x');
  Delta2=Delta2+delta3*A';
  Delta1=Delta1+delta2*x;
endfor
J= -k/m + (lambda*(sum(sum(theta1))+sum(sum(theta2))))/(2*m);
Theta1_grad(:,2:input_layer_size + 1)=Delta1(:,2:input_layer_size + 1)/m + (lambda*Theta1(:,2:input_layer_size + 1))/m;
Theta1_grad(:,1)=Delta1(:,1)/m;
Theta2_grad(:,1)=Delta2(:,1)/m;
Theta2_grad(:,2:hidden_layer_size + 1)=Delta2(:,2:hidden_layer_size + 1)/m + (lambda*Theta2(:,2:hidden_layer_size + 1))/m;

% --------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
