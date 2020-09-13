function [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_USPS(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3)
%% parameters

global C_old;
global train_per_class_old;
global test_per_class_old;

k=C*k_per_class; % number of dic columns
global n_new_old;
%n_new=540; % 100; % dimension of random faces

if(size(C_old,1)==0)
    C_old=C;
end
if(size(train_per_class_old,1)==0)
    train_per_class_old=train_per_class;
end
if(size(test_per_class_old,1)==0)
    test_per_class_old=test_per_class;
end
%% input data

global labels_all;
global X_all;
global labels_name_all;

global X;
global labels;
global labels_name;
global X_test;
global labels_test;
global labels_name_test;

srctrain='DBs\USPS\usps_train.txt';
X=importdata(srctrain);
X=X(1:end-1,:);
labels=X(:,1)';
labels=labels+1;
X=X(:,2:257)';

srctest='DBs\USPS\usps_test.txt';
X_test=importdata(srctest);
X_test=X_test(1:end-1,:);
labels_test=X_test(:,1)';
labels_test=labels_test+1;
X_test=X_test(:,2:257)';

n=size(X,1);
m=size(X,2);
m_test=size(X_test,2);

[labels labels_index]=sort(labels);
X=X(:,labels_index);

n_new=n;
labels_name=[];
labels_name_test=[];
data_refresh=0;

end

