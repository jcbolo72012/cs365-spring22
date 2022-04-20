data = [120 125 125 135 145 ; 61 60 64 68 72];

mean_vec = mean(data,2);
n = size(data,1)
B = data - mean_vec

S = (1/(n-1)) * B * transpose(B)

[U,L] = eig(S);

"Eigenvalues:"
l2 = L(1,1)
l1 = L(2,2)
"Eigenvectors:"
U

variance = l1/(l1+l2)

scatter(data(1,:),data(2,:))

