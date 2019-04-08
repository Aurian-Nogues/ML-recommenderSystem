function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% To be returned
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% X = movies x features (movie features)
% Y = movies x users (movie ratings)
% Theta = users x features (feature weights)
% R = movies x users (observations yes or no);

% collaborative filtering unregularized cost function
reviews = (X * Theta') .* R;
J = 1/2 * sum(sum((reviews - Y).^2));

%regularize cost function
reg = (lambda / 2) * (sum(sum(Theta.^2)) + sum(sum(X.^2)));
J = J + reg;

% calculate gradient of X (should be same size as X)
X_grad = ((X*Theta') .*R - Y) * Theta;
Theta_grad = ((X*Theta') .*R - Y)' * X;

% regularize gradient
X_grad = X_grad + lambda .* X;
Theta_grad = Theta_grad + lambda .* Theta;

grad = [X_grad(:); Theta_grad(:)];

end
