function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

ones = find(y == 1); 
zeroes = find(y == 0);

plot(X(ones,1),X(ones,2),'k+','markeredgecolor','r');
plot(X(zeroes,1),X(zeroes,2),'ko','markeredgecolor','b');







% =========================================================================



hold off;

end
