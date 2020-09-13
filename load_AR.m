function [C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh] = load_AR(C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3)
%% parameters
ishisteq=0;%0:no, 1:histeq
islbp=0;

global C_old;
global train_per_class_old;
global test_per_class_old;

k=C*k_per_class; % number of dic columns

global n_new_old;
n_new=540;%540;%300; % 100; % dimension of random faces
if(ishisteq)
	n_new=540;%1230;
end

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
global X_all_eq;
global X_all_eq2;
global X_all_lbp;

srcFolder='AR_warp\test2';
list=dir(srcFolder);
data_refresh=0;
if(size(X_all,1)==0 || n_new~=n_new_old)
    n_new_old=n_new;
    data_refresh=1;
    c=0;
    X_all=[];
    labels_all=[];
    labels_name_all={};
    for i=3:length(list)
        k1=strfind(list(i).name,'.bmp');
        if(size(k1,1)>0)
            l=str2double(list(i).name(3:5));
            if(list(i).name(1)=='W')
                l=l+50;
            end
            %list(i).name
            im=imread([srcFolder '\' list(i).name]);
            im=rgb2gray(im);
			im_eq=[];
			if(ishisteq)
				im_eq=histeq(im);
            end
            x_lbp=[];
            if(islbp)
                im2=imresize(im,.5);
                x_lbp=extractLBPFeatures(im2,'CellSize',[15 15],'Upright',0,'NumNeighbors',8,'Radius',1);
            end
            [s1 s2]=size(im);
            s=s1*s2;
            x=zeros(1,s);
			x_eq=x;
            % resample im to an array!
            for m=1:s1
                x((m-1)*s2+1:m*s2)=im(m,:);
            end
			
			if(ishisteq)
				% resample im_eq to an array!
				for m=1:s1
					x_eq((m-1)*s2+1:m*s2)=im_eq(m,:);
				end
			end
					
            x=x';
			x_eq=x_eq';
            x_lbp=x_lbp';
            c=c+1;
            X_all(:,c)=x;
			X_all_eq(:,c)=x_eq;
            %X_all_lbp(:,c)=x_lbp;
            labels_all(1,c)=l;
            labels_name_all{1,c}=[list(i).name];              
%             end
        end
    end          
end

%C=5;%length(list)-2; 
m=size(X_all,2);
n=size(X_all,1);
%============= generate random face descriptor ==========
if(n_new~=n_new_old || data_refresh)
    data_refresh=1;
    n_new_old=n_new;
    
    P=randn(n_new,n);
    for i=1:n_new
        P(i,:)=P(i,:)./norm(P(i,:),2);
    end
    X_all_new=P*X_all;
    X_all=X_all_new;
	
	if(ishisteq)
        X_all_eq_new=P*X_all_eq;
        
        %it also could be applied with different random vectors P, as follows:
        P2=randn(n_new,n);
        for i=1:n_new
            P2(i,:)=P2(i,:)./norm(P2(i,:),2);
        end
        X_all_eq_new2=P2*X_all_eq;
        
        X_all_eq=X_all_eq_new;
        X_all_eq2=X_all_eq_new2;
    end
    if(islbp)
        X_all_lbp=X_all_lbp;
    end
	
    n=n_new;
end

%==== drow random faces =======
if(0)
    n1=24;
    n2=21;
    im_rand=uint8(zeros(n1,n2));
    for i=1:m
        for j=1:n1
            im_rand(j,:)=X_all((j-1)*n2+1:j*n2,i);
        end
        im2=imresize(im_rand,8);
        imshow(im2);
        drawnow;
        pause(0.1);
        if(mod(i,64)==63)
            pause(3);
        end
    end
end
%=============select randomly 20 image of each person for train and the rest as test set ==============
global X;
global X_eq;
global X_eq2;
global X_lbp;
global labels;
global labels_name;
global X_test;
global X_test_eq;
global X_test_eq2;
global X_test_lbp;
global labels_test;
global labels_name_test;

if(data_refresh || train_per_class_old ~= train_per_class || test_per_class_old~=test_per_class || C_old~=C)
    train_per_class_old=train_per_class;
    test_per_class_old=test_per_class;
    C_old=C;
    data_refresh=1;
    
    X=[];
	X_eq=[];
    X_eq2=[];
    X_lbp=[];
    labels=[];
    labels_name={};

    X_test=[];
	X_test_eq=[];
    X_test_eq2=[];
    X_test_lbp=[];
    labels_test=[];
    labels_name_test={};
    for i=1:C
        inds=find(labels_all==i);
        rnds=inds(randperm(size(inds,2)));
        rnds_half1=rnds(1:train_per_class);%rnds(1:ceil(size(rnds,2)/13));
        if(test_per_class==-1)
            rnds_half2=rnds(train_per_class+1:end);
        else
            rnds_half2=rnds(train_per_class+1:train_per_class+test_per_class);%rnds(ceil(size(rnds,2)/13)+1:ceil(size(rnds,2)/13)*2);%ceil(size(rnds,2)/7)*2);%end);
        end
        X=[X X_all(:,rnds_half1)];
		if(ishisteq)
            X_eq=[X_eq X_all_eq(:,rnds_half1)];
            X_eq2=[X_eq2 X_all_eq2(:,rnds_half1)];
        end
        if(islbp)
            X_lbp=[X_lbp X_all_lbp(:,rnds_half1)];
        end
        labels=[labels labels_all(rnds_half1)];
        labels_name=[labels_name labels_name_all(rnds_half1)];

        X_test=[X_test X_all(:,rnds_half2)];
		if(ishisteq)
            X_test_eq=[X_test_eq X_all_eq(:,rnds_half2)];
            X_test_eq2=[X_test_eq2 X_all_eq2(:,rnds_half2)];
        end
        if(islbp)
            X_test_lbp=[X_test_lbp X_all_lbp(:,rnds_half2)];
        end
        labels_test=[labels_test labels_all(rnds_half2)];
        labels_name_test=[labels_name_test labels_name_all(rnds_half2)];
    end
end

m=size(X,2);
m_test=size(X_test,2);

[labels labels_index]=sort(labels);
X=X(:,labels_index);
if(ishisteq)
    X_eq=X_eq(:,labels_index);
    X_eq2=X_eq2(:,labels_index);
end
if(islbp)
    X_lbp=X_lbp(:,labels_index);
end

end

