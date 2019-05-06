function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vecC = [0.01 0.03 0.1 0.3 1 3 10 30];
vecS = [0.01 0.03 0.1 0.3 1 3 10 30];
minE = 1;

for i=1:length(vecC),
	for j=1:length(vecS),
		model = svmTrain(X, y, vecC(i), @(x1, x2) gaussianKernel(x1, x2, vecS(j)));
		pred = svmPredict(model, Xval);
		err = mean(double(pred ~= yval));
		if (err < minE),
			minE = err;
			C = vecC(i);
			sigma = vecS(j);
		end;
	end;
end;

fprintf('value of C: %f\nvalue of sigma: %f', C, sigma);
% =========================================================================

end
