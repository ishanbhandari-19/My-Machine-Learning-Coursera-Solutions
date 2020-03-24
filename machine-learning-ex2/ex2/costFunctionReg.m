function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n=size(theta)-1;
H=zeros(m,1);
K=0;
S=0;
for l=2:n+1,
  S=S+theta(l)^2;
endfor
for i=1:m,
H(i)=H(i)+ sigmoid(X(i,:)*theta);
K=K+y(i)*log(H(i))+(1-y(i))*log(1-H(i));
end;
J=-K/m +(lambda*S)/(2*m);
grad(1)=((H-y)'*X(:,1))/m;
for d=1:n,
  grad(d+1)=((H-y)'*X(:,d+1))/m +(lambda*theta(d+1))/m;
endfor
% =============================================================

end
