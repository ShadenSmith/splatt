cd ../csplatt/matlab/; make 
cd ../../tensor_toolbox;
addpath ../csplatt/matlab


X = sptensor(load_tt('/home/shaden/tensors/movielens.fixed.tns'));
XS = splatt_load('/home/shaden/tensors/movielens.fixed.tns');

T = tucker_als(X, 5);

%T.U{2}(1,:)
%T.U{3}(1,:)

Ys = tensor(splatt_ttm(XS, T.U, 1), [size(X,1), 5, 5]);
Yt = ttm(X, T.U, -1, 't');

Ys(1:2,:,:)
Yt(1:2,:,:)

