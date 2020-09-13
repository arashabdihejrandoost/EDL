function [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh]  = load_caltech101( C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3 )
if(1)
% %     This mat file contains spatial pyramid features for Caltech101 dataset. The dimension of each feature is 3000.
% % There are three variables in this file:
% % 
% % (1) featureMat:
% % A matrix of spatial pyramid features. Each column correspond to one spatial pyramid feature.
% % 
% % (2) filenameMat:
% % Image file names. Each cell correspond to the features from the same class in 'featureMat'. 
% % 
% % (3) labelMat:
% % This is a label matrix, each column corresponds to one spatial pyramid feature, where the non-zero position of each column 
% % indicates the class of the spatial pyramid feature.
    global featureMat;
    global filenameMat;
    global labelMat;
    if(size(featureMat,1)==0)
        load('LC-KSVD\Extracted Features\spatialpyramidfeatures4caltech101\spatialpyramidfeatures4caltech101.mat');
    end
end

%% parameters
global C_old;
global train_per_class_old;
global test_per_class_old;

C_all=size(labelMat,1);

k=C*k_per_class; % number of dic columns

n_new=size(featureMat,1);

refresh=0;
if(size(C_old,1)==0)
    C_old=C;
    refresh=1;
end
if(size(train_per_class_old,1)==0)
    train_per_class_old=train_per_class;
    refresh=1;
end
if(size(test_per_class_old,1)==0)
    test_per_class_old=test_per_class;
    refresh=1;
end
%% input data
%====== important: please normal data to have (for example) norm 2 value of one. ===========
global labels_all;
global X_all;
global labels_name_all;
X_all=featureMat;
labels_all=mod(find(labelMat),C_all)';
labels_all(labels_all==0)=C_all;
labels_name_all={};
for i=1:C_all
    a=filenameMat{1,i};
    s=size(a,2);
    for j=1:s
        labels_name_all{end+1:end+1} = a{1,j};  
    end    
end

m=size(X_all,2);
n=size(X_all,1);

%=============select randomly some of all data for train and the remailder as test set ==============
global X;
global labels;
global labels_name;
global X_test;
global labels_test;
global labels_name_test;
data_refresh=0;
if(refresh || train_per_class_old ~= train_per_class || test_per_class_old~=test_per_class || C_old~=C)
    train_per_class_old=train_per_class;
    test_per_class_old=test_per_class;
    C_old=C;
    data_refresh=1;
    
    X=[];
    labels=[];
    labels_name={};

    X_test=[];
    labels_test=[];
    labels_name_test={};
    for i=1:C
        inds=find(labels_all==i);
        rnds=inds(randperm(size(inds,2)));
        rnds_half1=rnds(1:train_per_class);%rnds(1:ceil(size(rnds,2)/13));
        if(test_per_class==-1)
            rnds_half2=rnds(train_per_class+1:end);
        else
            rnds_half2=rnds(train_per_class+1:min(train_per_class+test_per_class,end));%rnds(ceil(size(rnds,2)/13)+1:ceil(size(rnds,2)/13)*2);%ceil(size(rnds,2)/7)*2);%end);
        end
        X=[X X_all(:,rnds_half1)];
        labels=[labels labels_all(rnds_half1)];
        labels_name=[labels_name labels_name_all(rnds_half1)];

        X_test=[X_test X_all(:,rnds_half2)];
        labels_test=[labels_test labels_all(rnds_half2)];
        labels_name_test=[labels_name_test labels_name_all(rnds_half2)];
    end
end

m=size(X,2);
m_test=size(X_test,2);

[labels labels_index]=sort(labels);
X=X(:,labels_index);

end

