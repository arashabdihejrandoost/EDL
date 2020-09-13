function [train_err, test_err]=Entropy_Dic_Learning_ar(lambda1,lambda2,lambda3,b,skip_small_alphas,show_output)
%% 
% Entropy based Dictionary Learning (EDL), Version 1.0
% Copyright(c) 2020  Arash Abdi, Mohammad Rahmati, Mohammad Mahdi Ebadzadeh.
% All Rights Reserved.

% The code is for the paper: Arash Abdi, Mohammad Rahmati, Mohammad M.
% Ebadzadeh, Entropy based Dictionary Learning for Image Classification 

% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors.
% Contact: {arash_abdi,rahmati,ebadzadeh}@aut.ac.ir

%The most Important parameters are listed below, e.g. lambda1 and lambda2, k_per_class, train_per_class ,etc.

%% load data
DB='AR';
%DB='ExtendedYae';
%DB='caltech101'; 
C=100;
k_per_class=5;%number of columns for each class (would not assigned explicitly)
train_per_class=20;%number of training signals for each class
test_per_class=-1; % -1means the remainder...
use_omp_after_learn=1;%using omp to initialize A for cvx_grad() function. for using omp, columns of D should have L1_norm of one. so we normalize D in learning process (in each D update step). Although this speed up the process,it could increase the error ratio too.
sp_auto=0;%find best value for sp (sparsity threshold in OMP followed by cvx_grad() function). used in initializing A.
EDL2=0;%Entropy based Dictionary Learning Algorithm 2(using classification term in objective function)
savevar=0;%save variables on hard disk.
D_init_method=0;%0:from traning samples,1:from KSVD, 0<p<1: p percent from KSVD and the rest from traning samples, -1:no KSVD and no training samples.
%if D_init_method>0 ==> is_KSVD=1;
A_init_from_ksvd=0;
A_init_from_ksvd=A_init_from_ksvd & (D_init_method==1);
D_init_for_ksvd_rand=0;%1: random D for initilization of ksvd, 0:D init for ksvd from training data
ksvd_perClass=0;
Lasso_feature_sign=1;%1= feature sign, 0=our gradient based followed by OMP.
precision_in_each_loop=1;
sleepy=0;
mainloop_count_max=8;
if(~exist('skip_small_alphas'))
    skip_small_alphas=1;
end

if(Lasso_feature_sign)
    use_omp_after_learn=0;
    sp_auto=0;
end

if(~exist('lambda1'))
    lambda1=20
    lambda2=30e6
    lambda3=20000;
    b=1;
end

if(lambda3==0)
    EDL2=0;
end

if(~exist('show_output'))
    show_output=1;
end

[C,k_per_class,train_per_class,test_per_class,k,lambda1,lambda2,lambda3,n_new,n,m,m_test,X,labels,labels_name, X_test,labels_test,labels_name_test,data_refresh,back_ground_class] = load_data(DB,C,k_per_class,train_per_class,test_per_class,lambda1,lambda2,lambda3);
if(savevar)
    save('LastRunVars\X.mat','X');
    save('LastRunVars\labels.mat','labels');
    save('LastRunVars\X_test.mat','X_test');
    save('LastRunVars\labels_test.mat','labels_test');
end
%% initiate Dic and A(coefficiant)
addpath(genpath('sharingcode-LCKSVD\OMPbox'));
if(Lasso_feature_sign)
    addpath(genpath('sparse_coding'));
end
if(~exist('sp1'))
    if(sp_auto==0)
        sp1=k_per_class*12;%180;
        sp2=sp1;
    end
end
%-----------------------------
%===== let select some samplse as init D (with uniform sampling)======
k1=ceil((1-D_init_method)*k); %percent of dictionary from training samples
k2=k-k1;                      %percent of dictionary from KSVD
D1=zeros(n,k1);
D_inited=0;
if(k1>0)
    D_inited=1;
    s=m/k1;%floor(m/k);
    for i=1:k1
        D1(:,i)=X(:,floor((i-1)*s+1));
    end   
    %====== L2 normalizing columns of D =======
    for i=1:k1
        n2=norm(D1(:,i),2);              
        if(n2~=0)
            D1(:,i)=(D1(:,i)/n2);
        end
    end
end
%-----------------------------
D2=zeros(n,k2);
if(k2>1)                
    D_init=randn(n,k2);
    if(~D_init_for_ksvd_rand)
        s=m/k2;%floor(m/k);
        for i=1:k2
            D_init(:,i)=X(:,floor((i-1)*s+1));
        end
    end
    %====== L2 normalizing columns of D =======    
    for i=1:k2
        n2=norm(D_init(:,i),2);
        if(n2~=0)
            D_init(:,i)=(D_init(:,i)/n2);
        end
    end
    
    tic
    sp=floor(sp1/3);  
    [Dpart,Xpart,Errpart] = run_ksvd(DB,data_refresh,D_inited,X,D_init,labels,sp,ksvd_perClass);
    disp2('ksvd time:'); 
    toc;
    
    D2=Dpart;
end
%====merge D1 and D2 ======
D=[D1,D2];
global D_old;
D_refreshed=0;
if(size(D_old,1)==0 || size(D_old,2)~=size(D,2) || sum(sum(D_old~=D))>0)
    D_old=D;
    D_refreshed=1;
end
%========================= initial A ===================================
if(Lasso_feature_sign)
    beta = 1e-4;
    A_FS = D'*D + 2*beta*eye(size(D,2));
    Q = -D'*X;
end
if(1)
if(A_init_from_ksvd)
    %D=randn(n,k);
    A=zeros(k,m);
    A(:,:)=Xpart(:,:);
    %A=randn(k,m);
else
    %======= let set A with solving the "||DA-X||_2^2+y1.sum(||A_.j||_1)" =======
    global A_init;
    global lambda1_old;
    if(D_refreshed || size(lambda1_old,1)==0 || lambda1_old~=lambda1 || size(A_init,1)~=k || data_refresh)% || train_per_class_old ~= train_per_class || test_per_class_old~=test_per_class || C_old~=C)
        tic;
        lambda1_old=lambda1;
        C_old=C;
        
        A=zeros(k,m);
        if(~Lasso_feature_sign)
            sp=sp1;
            ompparams = {'checkdict','off','messages',-1};
            Gamma = omp(D'*X,D'*D,sp,ompparams{:});           
            AA=zeros(k,m);
            AA(:,:)=Gamma; 
        end
        tic;
        for i=1:m
            if(Lasso_feature_sign)
                a_grad = L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            else
                a_grad=cvx_grad(D,X(:,i),lambda1,0,1,AA(:,i));
            end
            A(:,i)=a_grad;
            if(mod(i,300)==0 && show_output)
                [-1 i]
                toc
            end
        end
        
        A_init=A;
    else
        A=A_init;
    end
end
else
    A=abs(randn(k,m));%.*(rand(k,m)<(sp1/k));
end
if(savevar)
    save('LastRunVars\A_init.mat','A');
    save('LastRunVars\D_init.mat','D');
end

%% main loop: find A and D respectly in a repeatitive manner.
mainloop_count=0;
all_count=0;
tic;
pcs=compute_pcs(A,labels,C);
rep=norm(D*A-X,'fro')^2;
spars=lambda1*sum(sum(abs(A)));
entropy=lambda2*ent_sum(pcs);
all=rep+spars+entropy;
if(EDL2)
    class_err=lambda3*err_sum(pcs'*abs(A),labels,b,m,C);%(A,pcs,labels,b);%(pcs'*abs(A),labels,b,m,C);
    all=all+class_err;
end
D_history=D;
if(show_output)
    h_opt=figure('name','optimization process');
    hold on;
    plot(all,'-k');
    plot(rep,'-g');
    plot(spars,'-b');
    plot(entropy,'-r');
    if(EDL2)
        plot(class_err,'-y');
    end

    disp2(strcat('representation norm (before learn) = ',num2str(norm(D*A-X,'fro')^2)));
    disp2(strcat('sparsity (before learn) = ',num2str(sum(sum(abs(A))))));
    disp2(strcat('Entropy on A_train(before learn) = ',num2str(ent_sum(pcs))));
    [mincolumn minclass]=min(sum(pcs));
    [maxcolumn maxclass]=max(sum(pcs));
    disp2('class with lowest/most D column before learn:');
    disp2(['class/conumn num= lowest: ' num2str(minclass) ' / ' num2str(mincolumn) '    most: ' num2str(maxclass) ' / ' num2str(maxcolumn)]);
end
if(precision_in_each_loop)  
    err_plot_train=[];
    err_plot_test=[];
    Ent_hist=[];
    h=figure('name','precision');
    title('precision');
    xlabel('iterations');
    ylabel('precision(%)');
    hold on;
    drawnow;
    h2=figure('name','Entropy');
    title('Entropy');
    xlabel('iterations');
    ylabel('sum of entropy');
    hold on;
    drawnow;    
    %% =========== compute  test error  to decide if continue the main loop   or not =====     
       %======== compute mu of D columns to each classe ==========
        %mu_all=zeros(k,C)
        d=1;
        mu_all=compute_pcs(A,labels,C); 
        Ent_hist(d)=ent_sum(mu_all);
        %========= classification on train data ===========    
        A_train=A;

        %fuzzy classify train data accordings to the mu of train data to each D column
        err=0;
        mu_train_to_classes=zeros(m,C);
        err_index=zeros(1,m);
        for i=1:m
            a=A_train(:,i);
            for j=1:k
                mu_train_to_classes(i,:)=mu_train_to_classes(i,:)+mu_all(j,:)*abs(a(j));
            end
            [val,c_ind]=max(mu_train_to_classes(i,:));
            if(c_ind~=labels(i))
                err=err+1;
                err_index(i)=1;
            end
        end
        terr_rate=err/m;
        err_plot_train(d)=100-terr_rate*100;

        %========= classification on test data ===========
        %compute A coefficiant for test data (without entropy term) as the mu of test data to each D column
        A_test=zeros(k,m_test);
        class_test=zeros(1,m_test);
        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X_test;
        for i=1:m_test
            a=L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            A_test(:,i)=a;  
        end
        
        %fuzzy classify test data accordings to the mu of train data to each D column
        err=0;
        mu_test_to_classes=zeros(m_test,C);
        err_index=zeros(1,m_test);
        for i=1:m_test
            a=A_test(:,i);
            for j=1:k
                mu_test_to_classes(i,:)=mu_test_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
            end
            [val,c_ind]=max(mu_test_to_classes(i,:));
            if(c_ind~=labels_test(i))
                err=err+1;
                err_index(i)=1;
            end
        end
        err_rate=err/m_test;
        err_plot_test(d)=100-err_rate*100; 
        disp2(['err on D_init=' num2str(err_rate)]);
        
        
        test_err=err_rate;
        train_err=terr_rate;

        h=figure(h);
        hold on;    
        plot(err_plot_train,'--g','LineWidth',2);
        plot(err_plot_test,'-b','LineWidth',2);
        drawnow;
        h2=figure(h2);
        plot(Ent_hist,'-r','LineWidth',2);    
        
        best_D=D;
        best_A=A_train;
        best_A_test=A_test;
end

%% main loop
pcs=compute_pcs(A,labels,C); %pi s of all classes in all rows of A. size= k,C
funcval1=norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*ent_sum(pcs);
%funcval1_prev_mainloop=10^100;
funcval1_prev_mainloop=funcval1;
tic;
while (mainloop_count<mainloop_count_max)
    %% find A by gradient decent
    %====== gradient for (||DA-X||_2^2+y1.sum(||A_.j||_1)) +y2.sum(H(A_i.)) ======      
    
    pcs=compute_pcs(A,labels,C); %pi s of all classes in all rows of A. size= k,C
    funcval1=norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*ent_sum(pcs);
    if(EDL2)
        funcval1=funcval1+lambda3*err_sum(pcs'*abs(A),labels,b,m,C);%(A,pcs,labels,b);%(pcs'*abs(A),labels,b,m,C);
    end    
    
    mainloop_count=mainloop_count+1;
    if(show_output)
        disp2('_____________'); 
        disp2(strcat('mainloop_count = ',num2str(mainloop_count)));
    end    
    
    step=0.1;%0.01;
    count=0;
    count_better=0;
    worse=0;
    der_size=1000;

    jump_count=0;
    jump_step=mean(mean(abs(A)))/10;%/2;%7;%==== important!
    local_mins=[];
    funcval1_prev=10^100;
    
    partial_der=1;
    if(partial_der==1)
        D_pow2=D.^2;
    end
    while((partial_der==0 && count<300 && count_better<80) || (partial_der==1 && count<1 && count_better<m*k)) %<20               
        minimization_percent=(funcval1_prev-funcval1)/funcval1_prev;
        funcval1_prev=funcval1;
        %========== jumping out after convergense ===========
        if(0 && ((partial_der==1 && minimization_percent<0.005) || (partial_der==0 &&  step < 0.001)))%(der_size < 0.01 || step_what < 0.001)% ??? <0.001 %important: thresholds depend on dimensaion of the problem and it's complexness! ======== 
            jump_count=jump_count+1;
            if(jump_count>5)
                break;
            end
            local_mins(jump_count)=funcval1;
            %======= if preveouse convergense was better, come back to there ====
            if(jump_count>1 && funcval1>func_last_converg)
                A=A_last_converg;
                funcval1=func_last_converg;
            end
            A_last_converg=A;
            func_last_converg=funcval1;

            % % pcs_temp=compute_pcs(A,labels,C);
            % % ed_temp=ent_der(A,labels,C,pcs_temp);
            % % A=A+lambda2*ed_temp*0.05;

            r=rand(k,m)*2-1;
            
            r=r*jump_step; 
            A=A+r;
 
            pcs=compute_pcs(A,labels,C);
            funcval1=norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*ent_sum(pcs);

            jump_step=jump_step*0.99; %====== important: rate of decreasing jump step (temperature) in a simulated anealing manner =======
            worse=0;
            step=0.01;       

        end
        count=count+1;
                       
        if(partial_der)   
            change_treshold=mean(mean(abs(A)))/100;% /100
            funcval1_min_process=zeros(k,m);
            %tic;
            norm_fro_elements_base=(D*A-X);
            norm_fro_pow2_base=sum(sum(norm_fro_elements_base.^2));
            A_abs_base=sum(sum(abs(A)));
            sum_A_rows=sum(abs(A),2);    
            sum_A_classes=pcs.*repmat(sum_A_rows,1,C);
            
            for i=1:k
                if (mod(i,1000)==0 && show_output)
                    disp2(strcat('A row number = ',num2str(i)));
                    disp2(strcat('target func value = ',num2str(funcval1)));
                    toc;
                end
                ent_sum_no_ij=ent_sum(pcs([1:i-1, i+1:end],:));%could be computed by just computing 2 rows of pcs:last [probably] changed row and the current excludung row. (for optimizing time consumint)
                if(EDL2)
                    if(i==1)
                        pcsT_dot_absA_no_ij=pcs([1:i-1, i+1:end],:)'*abs(A([1:i-1, i+1:end],:));
                    else
                        pcsT_dot_absA_no_ij=pcsT_dot_absA_no_ij+pcs(i-1,:)'*abs(A(i-1,:))-pcs(i,:)'*abs(A(i,:));%;pcsT_dot_absA_i_next;%pcs(i,:)'*abs(A(i,:));
                    end
%                     if(i~=k)
%                         pcsT_dot_absA_i_next=pcs(i+1,:)'*abs(A(i+1,:));
%                     end
                end
                for j=1:m
                    if(sleepy)
                        sleep();
                    end
                    %==== ignoring very small A_ij s in updateprocess results in saving time (about 20 times faster) and saving sparsity too!======= :
%                     if(A(i,j)<change_treshold)
%                         continue;
%                     end
                    if(skip_small_alphas && abs(A(i,j))<change_treshold)
                        continue;
                    end
                    step=0.1;
                    count_part=0;
                    
                    Azero_ij=A(i,j);
                    Azero_ij(Azero_ij==0)=1;
                    ed_ij=ent_part_der(A,labels,C,pcs,i,j);
                    der_ij=2*((D(:,i)'*D)*A(:,j)-D(:,i)'*X(:,j))+lambda1*A(i,j)/abs(Azero_ij)+lambda2*ed_ij;
                    if(EDL2)
                        pcsT_dot_absA=pcsT_dot_absA_no_ij+pcs(i,:)'*abs(A(i,:));
                        erd=err_part_der(pcsT_dot_absA,sum_A_rows(i),A,labels,C,pcs,i,j,b);                            
                        der_ij=der_ij+lambda3*erd;
                    end
                    der_ij_size=der_ij^2;                                       
                    
                    %ent_sum_no_ij=ent_sum(pcs([1:i-1, i+1:end],:));
                    while(count_part<10 && step > 0.06)  
                        count_part=count_part+1;
                        
                        A_ij_prev=A(i,j);
                        pc_row_prev=pcs(i,:);
                        ed_ij_prev=ed_ij;
                        der_ij_prev=der_ij;
                        der_ij_size_prev=der_ij_size;
                        sum_A_rows_i_prev=sum_A_rows(i);%%%
                        sum_A_classes_ic_prev=sum_A_classes(i,labels(j));%%%
                        
                        %=== jump -->
                        A(i,j)=A(i,j)-step*der_ij;
                        
                        %======== it could be the calculation of just modified A_ij =========== 
                        %======== and other elemnts could be get from previouse.    ===========                                           
                        %norm_fro_pow2=norm(D*A-X,'fro')^2;
                        
                        dif=(A(i,j)-A_ij_prev)*D(:,i);
                        dif_pow2=(A(i,j)-A_ij_prev)^2*D_pow2(:,i);
                        toAdd_elemnts=dif_pow2+2*dif.*norm_fro_elements_base(:,j);%+
                        norm_fro_pow2=norm_fro_pow2_base+sum(toAdd_elemnts);                        
                        dif_abs=abs(A(i,j))-abs(A_ij_prev);
                        A_abs=A_abs_base+dif_abs;
                                                                        
                        sum_A_classes(i,labels(j))=sum_A_classes(i,labels(j))+dif_abs;%%%
                        sum_A_rows(i)=sum_A_rows(i)+dif_abs;%%%                        
                        if(sum_A_rows(i)==0)%%%
                            sum_A_rows(i)=1;%%%
                        end%%%
                        pc_row=sum_A_classes(i,:)/sum_A_rows(i);%%%
                        %pc_row=compute_pc_row(A,labels,C,i); %%%
                        pcs(i,:)=pc_row;%%%
                        %funcval2=norm_fro_pow2+lambda1*sum(sum(abs(A)))+lambda2*(ent_sum_no_ij+ent(pc_row,1));%norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*(ent_sum_no_ij+ent(pc_row,1));%ent_sum(pcs);
                        funcval2=norm_fro_pow2+lambda1*A_abs+lambda2*(ent_sum_no_ij+ent(pc_row,1));
                        if(EDL2)        
                            pcsT_dot_absA=pcsT_dot_absA_no_ij+pcs(i,:)'*abs(A(i,:));
                            funcval2=funcval2+lambda3*err_sum(pcsT_dot_absA,labels,b,m,C);%(A,pcs,labels,b);%(pcsT_dot_absA,labels,b,m,C);%lambda3*err_sum(A,pcs,labels,b);
                        end
                        if(funcval2<funcval1)
                            count_better=count_better+1;
                            worse=0;
                            step=step*1.9; %===== important: rate of increasing step size for faster convergence =======
                            funcval1=funcval2;
                            
                            norm_fro_pow2_base=norm_fro_pow2;
                            norm_fro_elements_base(:,j)=norm_fro_elements_base(:,j)+dif;%toAdd_elemnts;
                            A_abs_base=A_abs;
                        else
                            worse=1;
                            step=step*0.4; %===== important: rate of decreasing step size for convergence, if we exceed the local min =======
                            A(i,j)=A_ij_prev;
                            pcs(i,:)=pc_row_prev;
                            ed_ij=ed_ij_prev;
                            der_ij=der_ij_prev;
                            der_ij_size=der_ij_size_prev;
                            sum_A_rows(i)=sum_A_rows_i_prev;%%%
                            sum_A_classes(i,labels(j))=sum_A_classes_ic_prev;%%%
                        end                    
                    end
                    funcval1_min_process(i,j)=funcval1;  
                end                
            end
            if(show_output)
                disp2(strcat('target func value (afetr whole A update) = ',num2str(funcval1)));
                toc
            end
        else        
            if(worse==0)            
                Azero=A;
                Azero(Azero==0)=1;
                ed=ent_der(A,labels,C,pcs);
                der=2*((D'*D)*A-D'*X)+lambda1*A./abs(Azero)+lambda2*ed;
                der_size=norm(der,'fro');

                A_prev=A;
                pcs_prev=pcs;
                ed_prev=ed;
                der_prev=der;
                der_size_prev=der_size;
            end

            A=A-step*der;
            pcs=compute_pcs(A,labels,C);
            funcval2=norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*ent_sum(pcs);
            if(funcval2<funcval1)
                count_better=count_better+1;
                worse=0;
                step=step*1.9; %===== important: rate of increasing step size for faster convergence =======
                funcval1=funcval2;
            else
                worse=1;
                step=step*0.7; %===== important: rate of decreasing step size for convergence, if we exceed the local min =======
                A=A_prev;
                pcs=pcs_prev;
                ed=ed_prev;
                der=der_prev;
                der_size=der_size_prev;
            end
        end

    end
    
    %======= if preveouse convergense was better, come back to there ====
    if(jump_count>0 && funcval1>func_last_converg)
        A=A_last_converg;
        funcval1=func_last_converg;
    end
    

    %% find D by minimizing ||DA-X||_2^2 s.t. for all i:||d_i||_2 <=1    %psudo inverse
    DD=X*pinv(A); %should check if size of columns exceed normal sizes!
    if(use_omp_after_learn)
        for i=1:k
            n2=norm(DD(:,i),2);
            if(n2~=0)
                DD(:,i)=DD(:,i)/n2;
            end
        end
    end
    D=DD;
    D_history(:,:,end+1)=D(:,:);
    %% ==== if no optimizataion after A and D update: do not continue;=====
    pcs=compute_pcs(A,labels,C); %pi s of all classes in all rows of A. size= k,C
    funcval1=norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*ent_sum(pcs);
    if(EDL2)
        funcval1=funcval1+lambda3*err_sum(pcs'*abs(A),labels,b,m,C);%(A,pcs,labels,b);%(pcs'*abs(A),labels,b,m,C);
    end    
    minimization_percent_mainloop=(funcval1_prev_mainloop-funcval1)/funcval1_prev_mainloop;
    funcval1_prev_mainloop=funcval1;
    if(~precision_in_each_loop && minimization_percent_mainloop<0.005)
        break;        
    end
    %% ============ show_output ==========================================
    if(show_output)
        disp2('_____________________');
    end
    %disp2('all_count,main loop, local means:');
    %all_count=all_count+count
    %mainloop_count
    %local_mins
    %toc;
    
    if(show_output)
        %figure;
        %plot(local_mins);
        pcs=compute_pcs(A,labels,C);
        rep(end+1)=norm(D*A-X,'fro')^2;
        spars(end+1)=lambda1*sum(sum(abs(A)));
        entropy(end+1)=lambda2*ent_sum(pcs);
        all(end+1)=rep(end)+spars(end)+entropy(end);
        if(EDL2)
            class_err(end+1)=lambda3*err_sum(pcs'*abs(A),labels,b,m,C);%(pcs'*abs(A),labels,b,m,C);%(A,pcs,labels,b);
            all(end)=all(end)+class_err(end);
        end
        
        h_opt=figure(h_opt);
        hold on;
        plot(all,'-k');
        plot(rep,'-g');
        plot(spars,'-b');
        plot(entropy,'-r');
        if(EDL2)
            plot(class_err,'-y');
        end
        drawnow;
    end
    %% ============ solve LASSO again for better initiated A ater D update =============
    if(Lasso_feature_sign)
        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X;
        A_new=zeros(k,m);
        for i=1:m
            a_grad = L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            A_new(:,i)=a_grad;
        end
        A=A_new;
    end
    %% =========== compute  test error  to decide if continue the main loop   or not =====
    if(precision_in_each_loop)      
       %======== compute mu (TaAllogh) of D columns to each classe ==========
        %mu_all=zeros(k,C)
        d=mainloop_count+1;
        mu_all=compute_pcs(A,labels,C); 
        Ent_hist(d)=ent_sum(mu_all);
        %========= classification on train data ===========    
        A_train=A;

        %fuzzy classify train data accordings to the mu of train data to each D column
        err=0;
        mu_train_to_classes=zeros(m,C);
        err_index=zeros(1,m);
        for i=1:m
            a=A_train(:,i);
            for j=1:k
                mu_train_to_classes(i,:)=mu_train_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
            end
            [val,c_ind]=max(mu_train_to_classes(i,:));
            if(c_ind~=labels(i))
                err=err+1;
                err_index(i)=1;
            end
        end
        terr_rate=err/m;
        err_plot_train(d)=100-terr_rate*100;

        %========= classification on test data ===========
        %compute A coefficiant for test data (without entropy term) as the mu of test data to each D column
        A_test=zeros(k,m_test);
        class_test=zeros(1,m_test);
        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X_test;
        for i=1:m_test
            a=L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            A_test(:,i)=a;  
        end
        
        %fuzzy classify test data accordings to the mu of train data to each D column
        err=0;
        mu_test_to_classes=zeros(m_test,C);
        err_index=zeros(1,m_test);
        for i=1:m_test
            a=A_test(:,i);
            for j=1:k
                mu_test_to_classes(i,:)=mu_test_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
            end
            [val,c_ind]=max(mu_test_to_classes(i,:));
            if(c_ind~=labels_test(i))
                err=err+1;
                err_index(i)=1;
            end
        end
        err_rate=err/m_test;
        err_plot_test(d)=100-err_rate*100;    
        disp2(['------ err on loop ' num2str(mainloop_count) ' = ' num2str(err_rate)]);

        h=figure(h);
        hold on;    
        plot(err_plot_train,'--g','LineWidth',2);
        plot(err_plot_test,'-b','LineWidth',2);
        drawnow;
        h2=figure(h2);
        plot(Ent_hist,'-r','LineWidth',2);
        drawnow;
        if(show_output)
            figure(h_opt);
            drawnow;
        end
        
        if(err_rate<=test_err)
            test_err=err_rate;
            train_err=terr_rate;
            best_D=D;
            best_A=A_train;
            best_A_test=A_test;
        elseif(strcmp(DB,'caltech101'))
            break;
        else
            ;
        end
    end
end
if(show_output)
    toc
end
if(savevar)
    save('LastRunVars\D_after.mat','D');
end

%% show best output 
if(precision_in_each_loop==1 && mainloop_count>0 && show_output)
    
    mu_all=compute_pcs(best_A,labels,C);

    disp2(strcat('representation norm (After learn) = ',num2str(norm(best_D*best_A-X,'fro')^2)));
    disp2(strcat('sparsity (After learn) = ',num2str(sum(sum(abs(best_A))))));
    disp2(strcat('Entropy on A_train(after learn) = ',num2str(ent_sum(mu_all))));
    [mincolumn, minclass]=min(sum(mu_all));
    [maxcolumn, maxclass]=max(sum(mu_all));
    disp2('class with lowest/most D column after learn:');
    disp2(['class/conumn num= lowest: ' num2str(minclass) ' / ' num2str(mincolumn) '    most: ' num2str(maxclass) ' / ' num2str(maxcolumn)]);


    disp2('on train:');
    disp2(train_err);

    disp2('on test:');
    disp2(test_err);   
else
%% start classification 
    %======= find A with solving the "||DA-X||_2^2+y1.sum(||A_.j||_1)" =======
    % we need this A to assign mu to D columns
    %it is possible to find this A with gradient decent, if it is faster!
    %A=zeros(k,m);


    if(Lasso_feature_sign)
        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X;
    end               
    if(precision_in_each_loop==1 && mainloop_count>0)
        AA=A;
    elseif(mainloop_count>0 || D_init_method==1)    
        AA=zeros(k,m);  
        if(use_omp_after_learn)
            for i=1:k
                n2=norm(D(:,i),2);
                if(n2~=0)
                    D(:,i)=(D(:,i)/n2);
                end
            end        

            sp=sp2;
            ompparams = {'checkdict','off','messages',-1};
            Gamma = omp(D'*X,D'*D,sp,ompparams{:});
            AG=zeros(k,m);        
            AG(:,:)=Gamma; 
        end
        AA=zeros(k,m);
        tic;
        for i=1:m
            if(Lasso_feature_sign)
                a_grad = L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            else 
                if(use_omp_after_learn)
                    a_grad=cvx_grad(D,X(:,i),lambda1,0,1,AG(:,i));
                else
                    a_grad=cvx_grad(D,X(:,i),lambda1,0,1);
                end
            end
            AA(:,i)=a_grad;
            if(mod(i,1000)==0 && show_output)
                [0 i]
                toc
            end
        end
    else 
        AA=A_init;
    end

    %======== compute mu of D columns to each classe ==========
    %mu_all=zeros(k,C)
    mu_all=compute_pcs(AA,labels,C); 

    %========= classification on train data ===========
    %compute A coefficiant for train data (without entropy term) as the mu of train data to each D column
    A_train=AA;

    if(show_output)
        disp2(strcat('representation norm (After learn) = ',num2str(norm(D*AA-X,'fro')^2)));
        disp2(strcat('sparsity (After learn) = ',num2str(sum(sum(abs(AA))))));
        disp2(strcat('Entropy on A_train(after learn) = ',num2str(ent_sum(mu_all))));
        [mincolumn minclass]=min(sum(mu_all));
        [maxcolumn maxclass]=max(sum(mu_all));
        disp2('class with lowest/most D column after learn:');
        disp2(['class/conumn num= lowest: ' num2str(minclass) ' / ' num2str(mincolumn) '    most: ' num2str(maxclass) ' / ' num2str(maxcolumn)]);
    end
    %fuzzy classify train data accordings to the mu of train data to each D column
    err=0;
    C_err=zeros(1,C);
    C_num=zeros(1,C);
    for i=1:C
        C_num(i)=sum(labels==i);
    end
    mu_train_to_classes=zeros(m,C);
    err_index=zeros(1,m);
    for i=1:m
        a=A_train(:,i);
        for j=1:k
            mu_train_to_classes(i,:)=mu_train_to_classes(i,:)+mu_all(j,:)*abs(a(j)); % ??? ===== abs(a(j))? if a column of D participate with negetive coeficiant in representation of a sample, we consider the abs of coefficiant as the contribution of that column in representation.
        end
        [val,c_ind]=max(mu_train_to_classes(i,:));
        if(c_ind~=labels(i))
            C_err(labels(i))=C_err(labels(i))+1;
            err=err+1;
            err_index(i)=1;
        end
    end
    C_err=C_err./C_num;
    err_rate=err/m;
    if(show_output)
        disp2('on train:');
        disp2(err_rate);
    end
    train_err=err_rate;
    if(savevar)
        save('LastRunVars\A_train.mat','A_train');
        save('LastRunVars\err_rate_on_train.mat','err_rate');
    end
    if(0)
        figure;
        plot(X(1,labels==1),X(2,labels==1),'b.')
        hold on;%linespec
        plot(X(1,labels==2),X(2,labels==2),'r.')
        plot(X(1,labels==3),X(2,labels==3),'y.')
        plot(X(1,labels==4),X(2,labels==4),'g.')
        plot(D(1,:),D(2,:),'mo') 

        plot(X(1,err_index==1),X(2,err_index==1),'ko');

    end

    %========= classification on test data ===========
    %compute A coefficiant for test data (without entropy term) as the mu of test data to each D column
    %A_test=zeros(k,m_test);
    class_test=zeros(1,m_test);

    if(Lasso_feature_sign)
        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X_test;
    end 
    if(use_omp_after_learn)
        sp=sp2;
        ompparams = {'checkdict','off','messages',-1};
        Gamma2 = omp(D'*X_test,D'*D,sp,ompparams{:});
        AA_test=zeros(k,m_test);
        AA_test(:,:)=Gamma2;
    end

    if(~(precision_in_each_loop==1 && mainloop_count>0))
        A_test=zeros(k,m_test);
        tic;
        for i=1:m_test
            if(Lasso_feature_sign)
                a_grad = L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
            else    
                if(use_omp_after_learn)
                    a_grad=cvx_grad(D,X_test(:,i),lambda1,0,1,AA_test(:,i));
                else
                    a_grad=cvx_grad(D,X_test(:,i),lambda1,0,1);
                end
            end
            %a_grad=Ent_Classify_grad(D,mu_all',X_test(:,i),lambda1,lambda3,0,1,AA_test(:,i));

            A_test(:,i)=a_grad;
            if(mod(i,1000)==0 && show_output)
                [1 i]
                toc
            end
        end
    end

    %fuzzy classify test data accordings to the mu of train data to each D column
    err=0;
    C_err_test=zeros(1,C);
    C_num_test=zeros(1,C);
    for i=1:C
        C_num_test(i)=sum(labels_test==i);
    end
    mu_test_to_classes=zeros(m_test,C);
    err_index=zeros(1,m_test);
    for i=1:m_test
        a=A_test(:,i);
        for j=1:k
            mu_test_to_classes(i,:)=mu_test_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
        end
        [val,c_ind]=max(mu_test_to_classes(i,:));
        if(c_ind~=labels_test(i))
            C_err_test(labels_test(i))=C_err_test(labels_test(i))+1;
            err=err+1;
            err_index(i)=1;
        end
    end
    C_err_test=C_err_test./C_num_test;
    err_rate=err/m_test;
    if(show_output)
        disp2('on test:');
        disp2(err_rate);
    end 
    if(precision_in_each_loop==0)
        test_err=err_rate;
    end
    if(savevar)
        save('LastRunVars\A_test.mat','A_test');
        save('LastRunVars\err_rate_on_test.mat','test_err');
    end

    precision_history=0;
    if(precision_history)
        s=size(D_history,3);
        err_plot_train=[];
        err_plot_test=[];
        Ent_hist=[];
        h=figure('name','precision');
        title('precision');
        xlabel('iterations');
        ylabel('precision(%)');
        hold on;
        drawnow;
        h2=figure('name','Entropy');
        title('Entropy');
        xlabel('iterations');
        ylabel('sum of entropy');
        hold on;
        drawnow;
        for d=1:s
            D=D_history(:,:,d);

        beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X;

            AA=zeros(k,m);
            for i=1:m
               a = L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
                AA(:,i)=a;
                if(show_output)
                    if mod(i,1000)==0
                        [d 0 i]
                    end
                end
            end

            %======== compute mu (TaAllogh) of D columns to each classe ==========
            %mu_all=zeros(k,C)
            mu_all=compute_pcs(AA,labels,C); 
            Ent_hist(d)=ent_sum(mu_all);
            %========= classification on train data ===========    
            A_train=AA;


            %fuzzy classify train data accordings to the mu of train data to each D column
            err=0;
            mu_train_to_classes=zeros(m,C);
            err_index=zeros(1,m);
            for i=1:m
                a=A_train(:,i);
                for j=1:k
                    mu_train_to_classes(i,:)=mu_train_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
                end
                [val,c_ind]=max(mu_train_to_classes(i,:));
                if(c_ind~=labels(i))
                    err=err+1;
                    err_index(i)=1;
                end
            end
            terr_rate=err/m;
            err_plot_train(d)=100-terr_rate*100;

            %========= classification on test data ===========
            %compute A coefficiant for test data (without entropy term) as the mu of test data to each D column
            A_test=zeros(k,m_test);
            class_test=zeros(1,m_test);

    beta = 1e-4;
        A_FS = D'*D + 2*beta*eye(size(D,2));
        Q = -D'*X_test;

            for i=1:m_test
                %a=zeros(k,1);
                a=L1QP_FeatureSign_yang(lambda1/2, A_FS, Q(:,i));
                A_test(:,i)=a;  
                if(show_output)
                    if mod(i,1000)==0
                        [d 1 i]
                    end
                end
            end


            %fuzzy classify test data accordings to the mu of train data to each D column
            err=0;
            mu_test_to_classes=zeros(m_test,C);
            err_index=zeros(1,m_test);
            for i=1:m_test
                a=A_test(:,i);
                for j=1:k
                    mu_test_to_classes(i,:)=mu_test_to_classes(i,:)+mu_all(j,:)*abs(a(j)); 
                end
                [val,c_ind]=max(mu_test_to_classes(i,:));
                if(c_ind~=labels_test(i))
                    err=err+1;
                    err_index(i)=1;
                end
            end
            err_rate=err/m_test;
            err_plot_test(d)=100-err_rate*100;

            if(err_rate<test_err)
                test_err=err_rate;
                train_err=terr_rate;
            end

            h=figure(h);
            hold on;    
            plot(err_plot_train,'--g','LineWidth',2);
            plot(err_plot_test,'-b','LineWidth',2);
            drawnow;
            h2=figure(h2);
            plot(Ent_hist,'-r','LineWidth',2);
            drawnow;
        end 
    end
end
end

function ers=err_sum(pcsT_dot_absA,labels,b,m,C)
%ers:error sum
%m=size(A,2);
ers=0;
%C=size(pcs,2);
%%pcsp_absA=pcs'*abs(A);%pcs_prim×abs_A

R=zeros(C,m);
for j=1:m
    R(:,j)=pcsT_dot_absA(labels(j),j);
end
ers=sum(log(sum(exp((pcsT_dot_absA(:,:)-R)*b))));
end

function dsm=dis_soft_max(A,pcs,j,cs,C,b,pcs_prim_abs_A)%discriminative soft max
s=0;
%abs_A_j=abs(A(:,j));
M_csj=pcs_prim_abs_A(cs,j);%pcs(:,cs)'*abs_A_j;
for c=1:C    
    M_cj=pcs_prim_abs_A(c,j);%pcs(:,c)'*abs_A_j;    
    s=s+exp((M_cj-M_csj)*b);
end
dsm=log(s);
end

function erd=err_part_der(pcsT_dot_absA,sum_absA_rows_ip,A,labels,C,pcs,ip,jp,b)
m=size(A,2);
erd=0;
abs_A=abs(A);
%=======compute and save c dependent(and j indipendent) values to prevent recomputing them.=======
ro_pic_Aij_All=zeros(1,C);
for c=1:C
    ro_pic_Aij_All(c)=ro_pic_Aij(sum_absA_rows_ip,A,labels,ip,jp,c);
end

pcsp_absA=pcsT_dot_absA;%pcs'*abs_A;%pcs_prim×abs_A
for j=1:m
    cs=labels(j);%c_star

    if(A(ip,j)>0)
        coef=1;
    elseif(A(ip,j)<0)
        coef=-1;
    else
        coef=0;
    end
    
    %abs_A_j=abs(A(:,j));
    %M_csj=pcsp_absA(cs,j);%pcs(:,cs)'*abs_A_j;
    t3=0;
    t2_sum=0;
    for c=1:C
        t1=0;
        if(c~=cs)            
            %ro1=ro_pic_Aij(A,labels,ip,jp,c);
            ro1=ro_pic_Aij_All(c);
            %ro2=ro_pic_Aij(A,labels,ip,jp,cs);
            ro2=ro_pic_Aij_All(cs);
            t1=(ro1-ro2)*abs_A(ip,j);%abs(A(ip,j));
                
            if(jp==j)
                t1=t1+coef*(pcs(ip,c)-pcs(ip,cs));%(A(ip,j)/abs_Aipj_nozero)*(pcs(ip,c)-pcs(ip,cs));
            end
        end
        t1=t1*b;
                
        %M_cj=pcsp_absA(c,j);%pcs(:,c)'*abs_A_j;       
        t2=exp((pcsp_absA(c,j)-pcsp_absA(cs,j))*b);%exp((M_cj-M_csj)*b);
        t2_sum=t2_sum+t2;
        
        t3=t3+t1*t2;
    end    
    
    erd=erd+(t3/t2_sum);
end
end

function es=ent_sum(pcs)
k=size(pcs,1);
es=0;
for i=1:k
    es=es+ent(pcs,i);
end
end

function E=ent(pcs,row)
C=size(pcs,2);
if(C==1)
    C=2;
end
pc=pcs(row,:);
pc2=pc;
pc2(pc2==0)=1;
E=-1*sum(pc.*logb(pc2,C));
end

function ed=ent_der(A,labels,C,pcs)
[k,m]=size(A);
ed=zeros(k,m);
for i=1:k
    row_sum_2=sum(abs(A(i,:)))^2;
    if(row_sum_2==0)
        row_sum_2=1;
    end
    for j=1:m
        if(A(i,j)==0)
            Aij_nozero=1;
        else
            Aij_nozero=A(i,j);
        end
        for c=1:C  
            classmate=0;
            if(labels(j)==c)
                classmate=1;
            end
            if(classmate)
                
                ro_pic_Aij_in=(sum(abs(A(i,labels~=c)))/row_sum_2)*(A(i,j)/abs(Aij_nozero));
                %===compute logb(pcs(i,c),C), not to call function for time consumings!
                if(pcs(i,c)==0)
                    l=0;
                else
                    l=log2(pcs(i,c))/log2(C);
                end
                %l=logb(pcs(i,c),C);
                ed(i,j)=ed(i,j)-(ro_pic_Aij_in*l+(ro_pic_Aij_in/log(C))); %(ro_pic_Aij_in/(pcs(i,c)*log(C))) * pcs(i,c));
            else
                ro_pic_Aij_out=(-sum(abs(A(i,labels==c)))/row_sum_2)*(A(i,j)/abs(Aij_nozero));
                %===compute logb(pcs(i,c),C), not to call function for time consumings!
                if(pcs(i,c)==0)
                    l=0;
                else
                    l=log2(pcs(i,c))/log2(C);
                end
                %l=logb(pcs(i,c),C);
                ed(i,j)=ed(i,j)-(ro_pic_Aij_out*l + (ro_pic_Aij_out/log(C)));%(ro_pic_Aij_out/(pcs(i,c)*log(C))) * pcs(i,c));
            end
        end
    end
end
end

% ==== just one dimension derivation ========
function ed=ent_part_der(A,labels,C,pcs,i,j) %partial der
%[k,m]=size(A);
ed=zeros(1,1);%(k,m);

row_sum_2=sum(abs(A(i,:)))^2;
if(row_sum_2==0)
    row_sum_2=1;
end

if(A(i,j)==0)
    Aij_nozero=1;
else
    Aij_nozero=A(i,j);
end

for c=1:C  
    classmate=0;
    if(labels(j)==c)
        classmate=1;
    end
    if(classmate)

        ro_pic_Aij_in=(sum(abs(A(i,labels~=c)))/row_sum_2)*(A(i,j)/abs(Aij_nozero));
        %===compute logb(pcs(i,c),C), not to call function for time consumings!
        if(pcs(i,c)==0)
            l=0;
        else
            l=log2(pcs(i,c))/log2(C);
        end
        %l=logb(pcs(i,c),C);
        ed=ed-(ro_pic_Aij_in*l+(ro_pic_Aij_in/log(C))); %(ro_pic_Aij_in/(pcs(i,c)*log(C))) * pcs(i,c));
    else
        ro_pic_Aij_out=(-sum(abs(A(i,labels==c)))/row_sum_2)*(A(i,j)/abs(Aij_nozero));
        %===compute logb(pcs(i,c),C), not to call function for time consumings!
        if(pcs(i,c)==0)
            l=0;
        else
            l=log2(pcs(i,c))/log2(C);
        end
        %l=logb(pcs(i,c),C);
        ed=ed-(ro_pic_Aij_out*l + (ro_pic_Aij_out/log(C)));%(ro_pic_Aij_out/(pcs(i,c)*log(C))) * pcs(i,c));
    end
end

end

function r=ro_pic_Aij(sum_absA_rows_ip,A,labels,i,j,c)
row_sum_2=sum_absA_rows_ip^2;%sum(abs(A(i,:)))^2;
if(row_sum_2==0)
    row_sum_2=1;
end
if(A(i,j)>0)
        coef=1;
elseif(A(i,j)<0)
    coef=-1;
else
    coef=0;
end
% if(A(i,j)==0)
%     Aij_nozero=1;
% else
%     Aij_nozero=A(i,j);
% end
classmate=0;
if(labels(j)==c)
    classmate=1;
end

if(classmate)
    r=(sum(abs(A(i,labels~=c)))/row_sum_2)*coef;%(A(i,j)/abs(Aij_nozero));   
else
    r=(-sum(abs(A(i,labels==c)))/row_sum_2)*coef;%(A(i,j)/abs(Aij_nozero));    
end

end

function pcs=compute_pcs(A,labels,C)
[k,m]=size(A);
pcs=zeros(k,C);
for i=1:k
    s=sum(abs(A(i,:)));
    if(s==0)
        s=1;
    end
    for j=1:C
        pcs(i,j)=sum(abs(A(i,labels==j)))/s;
    end
end

end

function pcs=compute_pc_row(A,labels,C,i)
pcs=zeros(1,C);
s=sum(abs(A(i,:)));
if(s==0)
    s=1;
end
for j=1:C
    pcs(j)=sum(abs(A(i,labels==j)))/s;
end

end

function l=logb(x,b)
if(x==0)
    l=0;
else
    l=log2(x)/log2(b);
end
end

function sp=find_sp(D,X,lambda1,k_per_class,i)
k=size(D,2);
%i=2;%the column of X to check similarity of cvx coefficient ond grad coefficient for.
cvx_clear
cvx_begin quiet
cvx_solver sedumi
cvx_precision low
    variables a(k,1)
    minimize(pow_pos(norm(D*a-X(:,i),'fro'),2)+lambda1*norm(a,1));%(+lambda*norm(xx,1)+alpha*pow_pos(norm(Q(:,j)-A*xx,2),2))
cvx_end
t1=norm(D*a-X(:,i),'fro')^2;
t2=lambda1*sum(abs(a));
f=t1+t2;

ompparams = {'checkdict','off','messages',-1};
lb=k_per_class*3;
ub=min(k_per_class*30,k);
st=k_per_class*2;

ff_old=10^10;
for sp=lb:st:ub   
    Gamma = omp(D'*X(:,i),D'*D,sp,ompparams{:});
    A=zeros(k,1);
    A(:,:)=Gamma;
    A=cvx_grad(D,X(:,2),lambda1,0,1,A);
    
    tt1=norm(D*A-X(:,i),'fro')^2;
    tt2=lambda1*sum(abs(A));
    ff=tt1+tt2;
    
    if(abs(1-tt1/t1) < 0.01 && abs(1-tt2/t2) < 0.01)
        sp_best=sp;
        break;
    end
    
    if(ff<ff_old)
        ff_old=ff;
        sp_best=sp;
    end
end
sp=sp_best;
end

function sleep()
fileID = fopen('C:\sleep.txt');
C = textscan(fileID, '%d');
fclose(fileID);
run=C{1};
s=[];         
if(~run)
    if(evalin('caller','exist(''lambda1'')'))
        evalin('caller','save all.mat');
        %s = evalin('caller','dbstatus');
        evalin('caller','clear');
    end
end
while(~run)
    fileID = fopen('C:\sleep.txt');
    C = textscan(fileID, '%d');
    fclose(fileID);
    run=C{1};
    pause(10);    
end
if(evalin('caller','~exist(''lambda1'')'))    
    evalin('caller','load all.mat');
end
end

function disp2(o)
disp(o);
id=fopen('output.txt','a+');
fprintf(id, num2str(o));
fprintf(id, '\n');
fclose(id);
end

