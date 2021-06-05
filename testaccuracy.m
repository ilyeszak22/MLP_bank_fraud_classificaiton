function accuracy =testaccuracy(X , y , Theta1,Theta2)
m = size(X, 1);
a1=[ones(m,1),X];
a2=[ones(m,1),sigmoid(a1*Theta1')];
a3=sigmoid(a2*Theta2');
n=round(a3);
M=n==y;
accuracy=sum(M)/length(M);
end
