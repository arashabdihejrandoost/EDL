function [Dpart,Xpart,Errpart] = run_ksvd(DB,data_refresh,D_init,X,D,labels,sp,ksvd_perClass)
global Dpart;
global Xpart;
Errpart=-1;
global iterations_old;
global sparsitythres_old;
addpath(genpath('sharingcode-LCKSVD'));
if(strcmp(DB , 'AR') || strcmp(DB , 'ExtendedYale') || strcmp(DB , 'USPS'))
    iterations = 50; %for init ksvd
    sparsitythres = sp; % sparsity prior    
    if(data_refresh || size(D,2)~=size(Dpart,2) ||size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres || D_init==1)
        sparsitythres_old=sparsitythres;
        iterations_old=iterations;        
        para.data = X;
        para.Tdata = sparsitythres;
        para.iternum = iterations;
        para.memusage = 'high';
        % normalization
        para.initdict = D;
        % ksvd process
        [Dpart,Xpart,Errpart] = ksvd(para,'');
    end
elseif(strcmp(DB , 'caltech101'))
    perClass=0;
    if(perClass)
        iterations = 20; %for init ksvd
        %sparsitythres = 20; % sparsity prior  
        C=max(double(labels));
        k_perclass=floor(size(D,2)/C);
        sparsitythres = k_perclass;
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres || D_init==1)
 

            
            sparsitythres_old=sparsitythres;
            iterations_old=iterations; 
            
            Dpart=zeros(size(D));
            Xpart=zeros(size(D,2),size(X,2));
            for i=1:C
                lb=min(find(labels==i));
                ub=max(find(labels==i));
                c_num=ub-lb+1;
                X_c=X(:,lb:ub);
                D_c=D(:,(i-1)*k_perclass+1:i*k_perclass);
                para.data = X_c;
                para.Tdata = sparsitythres;
                para.iternum = iterations;
                para.memusage = 'high';
                % normalization
                para.initdict = D_c;
                % ksvd process
                [Dpart_c,Xpart_c,Errpart] = ksvd(para,''); 
                Dpart(:,(i-1)*k_perclass+1:i*k_perclass)=Dpart_c;
                Xpart((i-1)*k_perclass+1:i*k_perclass,lb:ub)=Xpart_c(:,:);
            end
        end
    else
        iterations = 50; %for init ksvd
        sparsitythres = 40; % 40 :sparsity prior    
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres) % || D_init==1)
            sparsitythres_old=sparsitythres;
            iterations_old=iterations;                    
            para.data = X;
            para.Tdata = sparsitythres;
            para.iternum = iterations;
            para.memusage = 'high';
            % normalization
            para.initdict = D;
            % ksvd process
            [Dpart,Xpart,Errpart] = ksvd(para,'');                
        end
    end
    
elseif(strcmp(DB , 'cifar10'))
    perClass=ksvd_perClass;
    C=max(double(labels));
    if(perClass)
        iterations = 4; %for init ksvd
        %sparsitythres = 20; % sparsity prior  
        
        k_perclass=floor((size(D,2)/C)/3);
        sparsitythres = k_perclass;
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres || D_init==1)
 

            
            sparsitythres_old=sparsitythres;
            iterations_old=iterations; 
            
            Dpart=zeros(size(D));
            Xpart=zeros(size(D,2),size(X,2));
            for i=1:C
                lb=min(find(labels==i));
                ub=max(find(labels==i));
                c_num=ub-lb+1;
                X_c=X(:,lb:ub);
                D_c=D(:,(i-1)*k_perclass+1:i*k_perclass);
                para.data = X_c;
                para.Tdata = sparsitythres;
                para.iternum = iterations;
                para.memusage = 'high';
                % normalization
                para.initdict = D_c;
                % ksvd process
                [Dpart_c,Xpart_c,Errpart] = ksvd(para,''); 
                Dpart(:,(i-1)*k_perclass+1:i*k_perclass)=Dpart_c;
                Xpart((i-1)*k_perclass+1:i*k_perclass,lb:ub)=Xpart_c(:,:);
            end
        end
    else
        iterations = 50; %for init ksvd
        sparsitythres = sp;% 40 :sparsity prior    
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres) % || D_init==1)
            sparsitythres_old=sparsitythres;
            iterations_old=iterations;                    
            para.data = X;
            para.Tdata = sparsitythres;
            para.iternum = iterations;
            para.memusage = 'high';
            % normalization
            para.initdict = D;
            % ksvd process
            [Dpart,Xpart,Errpart] = ksvd(para,'');                
        end
    end
elseif(strcmp(DB , 'flickr32'))
    perClass=ksvd_perClass;
    C=max(double(labels));
    if(perClass)
        iterations = 4; %for init ksvd
        %sparsitythres = 20; % sparsity prior  
        
        k_perclass=floor((size(D,2)/C)/1);
        sparsitythres = k_perclass;
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres || D_init==1)
 

            
            sparsitythres_old=sparsitythres;
            iterations_old=iterations; 
            
            Dpart=zeros(size(D));
            Xpart=zeros(size(D,2),size(X,2));
            for i=1:C
                lb=min(find(labels==i));
                ub=max(find(labels==i));
                c_num=ub-lb+1;
                X_c=X(:,lb:ub);
                D_c=D(:,(i-1)*k_perclass+1:i*k_perclass);
                para.data = X_c;
                para.Tdata = sparsitythres;
                para.iternum = iterations;
                para.memusage = 'high';
                % normalization
                para.initdict = D_c;
                % ksvd process
                [Dpart_c,Xpart_c,Errpart] = ksvd(para,''); 
                Dpart(:,(i-1)*k_perclass+1:i*k_perclass)=Dpart_c;
                Xpart((i-1)*k_perclass+1:i*k_perclass,lb:ub)=Xpart_c(:,:);
            end
        end
    else
        iterations = 50; %for init ksvd
        sparsitythres = sp;% 40 :sparsity prior    
        if(data_refresh || size(iterations_old,1)==0 || iterations_old~=iterations || sparsitythres_old~=sparsitythres) % || D_init==1)
            sparsitythres_old=sparsitythres;
            iterations_old=iterations;                    
            para.data = X;
            para.Tdata = sparsitythres;
            para.iternum = iterations;
            para.memusage = 'high';
            % normalization
            para.initdict = D;
            % ksvd process
            [Dpart,Xpart,Errpart] = ksvd(para,'');                
        end
    end
end
end

