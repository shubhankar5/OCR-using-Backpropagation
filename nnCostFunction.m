function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(m,1) X];
a1=X;
a2=sigmoid(a1*Theta1');
a2=[ones(m,1) a2];
a3=sigmoid(a2*Theta2');
hyp=a3;
for i=1:num_labels,
  num(i)=i;
endfor
new_y=y==num;
temp1=Theta1(:,2:size(Theta1,2));
temp2=Theta2(:,2:size(Theta2,2));
J=-(sum(sum(new_y.*log(hyp)+(1.-new_y).*log(1.-hyp))))/m + lambda*(sum(sum(temp1.*temp1))+sum(sum(temp2.*temp2)))/(2*m);
capdelta1=zeros(hidden_layer_size,size(a1,2));
capdelta2=zeros(num_labels,size(a2,2));
for i=1:m,
  a1=X(i,:);
  a2=sigmoid(a1*Theta1');
  a2=[1 a2];
  a3=sigmoid(a2*Theta2');
  for k=1:num_labels,
    a(k)=a3(k)-new_y(i,k);
  endfor
  delta3=a';
  delta2=(temp2'*delta3).*sigmoidGradient((a1*Theta1')');
  capdelta1=capdelta1+delta2*a1;
  capdelta2=capdelta2+delta3*a2;
endfor
temp1=[zeros(size(temp1,1),1) temp1];
temp2=[zeros(size(temp2,1),1) temp2];
D1=capdelta1/m + lambda*(temp1)/m;
D2=capdelta2/m + lambda*(temp2)/m;
Theta1_grad=D1;
Theta2_grad=D2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
