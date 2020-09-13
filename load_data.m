function [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh,background_class] = load_data( DB,C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3)
if(strcmp(DB , 'ExtendedYale'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_ExtendedYale(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=0;
elseif(strcmp(DB , 'caltech101'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_caltech101(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=1;
elseif(strcmp(DB , 'AR'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_AR(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=1;
elseif(strcmp(DB , 'USPS'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_USPS(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=1; 
elseif(strcmp(DB , 'cifar10'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_cifar10(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=1;
elseif(strcmp(DB , 'flickr32'))
    [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_flickr32(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
    background_class=1;
end

end

