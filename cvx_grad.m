function a = cvx_grad(D,x,lambda1,show_output,skip_small,a_init)
%% find A by gradient decent
[n k]=size(D);
%a=zeros(k,1);
if(~exist('a_init'))
    %predict huresticly a good initial a
    a1=pinv(D)*x;
    a2=zeros(k,1);
    fmin=10^100;
    for p=0:0.1:1
        a3=p*a2+(1-p)*a1;
        f=norm(D*a3-x,'fro')^2+lambda1*sum(abs(a3));
        if(f<fmin)
           fmin=f;
           amin=a3;
        end
    end
    a=amin;
else
    a=a_init;
end
%====== gradient for (||DA-X||_2^2+y1.sum(||A_.j||_1)) =====
funcval1=norm(D*a-x,'fro')^2+lambda1*sum(abs(a));
step=0.1;%0.01;
count=0;
count_better=0;
worse=0;
der_size=1000;

jump_count=0;
jump_step=mean(mean(abs(a)))/10;%/2;%7;%==== important!!!!!!
local_mins=[];
funcval1_prev=10^7;

partial_der=1;
if(partial_der==1)
    D_pow2=D.^2;
end
    
while((partial_der==0 && count<30000 && count_better<8000) || (partial_der==1 && count<10 && count_better<k*100)) %<20               
    minimization_percent=(funcval1_prev-funcval1)/funcval1_prev;
    if(minimization_percent<0.001)
        break;
    end
    funcval1_prev=funcval1;
%     %========== jumping out after convergense ===========
%     if((partial_der==1 && minimization_percent<0.005) || (partial_der==0 &&  step < 0.001))%(der_size < 0.01 || step_what < 0.001)% ??? <0.001 %important: thresholds depend on dimensaion of the problem and it's complexness! ======== 
%         jump_count=jump_count+1;
%         if(jump_count>5)
%             break;
%         end
%         local_mins(jump_count)=funcval1;
%         %======= if preveouse convergense was better, come back to there ====
%         if(jump_count>1 && funcval1>func_last_converg)
%             a=a_last_converg;
%             funcval1=func_last_converg;
%         end
%         a_last_converg=a;
%         func_last_converg=funcval1;
% 
%         r=rand(k,1)*2-1;
%         %r(abs(r)<0.7)=0; %====== important: rate of  random changing of A_ij s. it could be decrease slowly in a simulated anealing manner =======
%         r=r*jump_step; % ===== depending of the complexness of the problem, this amount of jumping(rate of changes and amount of them) could be very much! =====
%         a=a+r;
% 
%         % mm=mean(mean(A));
%         % A(abs(A)<mm*10)=0;
% 
%         funcval1=norm(D*a-x,'fro')^2+lambda1*sum(abs(a));
% 
%         jump_step=jump_step*0.99; %====== important: rate of decreasing jump step (temperature) in a simulated anealing manner =======
%         worse=0;
%         step=0.1;       
% 
%     end
    count=count+1;

    if(partial_der)   
        change_treshold=mean(abs(a))/100;% /100
        %funcval1_min_process=zeros(k,m);
        %tic;
        norm_fro_elements_base=(D*a-x);
        norm_fro_pow2_base=sum(norm_fro_elements_base.^2);
        a_abs_base=sum(abs(a));
%         sum_A_rows=sum(abs(a),2);    
%         sum_A_classes=pcs.*repmat(sum_A_rows,1,C);
%         ent_sum_no_ij=ent_sum(pcs([1:i-1, i+1:end],:));%could be computed by just computing 2 rows of pcs:last [probably] changed row and the current excludung row. (for optimizing time consumint)
% azero_all=a;
% azero_all(azero_all==0)=1;
% der_all=2*((D'*D)*a-D'*x)+lambda1*a./abs(azero_all);
%[so,soind]=sort(abs(der_all),'descend');
        for i=1:k
%             azero_all=a;
%             azero_all(azero_all==0)=1;
%             der_all=2*((D'*D)*a-D'*x)+lambda1*a./abs(azero_all);
%             [ma,maind]=max(abs(der_all));
            %i=maind;
            %i=ii;
            %i=soind(ii);%k
            %==== ignoring very small A_ij s in updateprocess results in saving time (about 20 times faster) and saving sparsity too!======= :
            if(skip_small && abs(a(i))<change_treshold)
                continue;
            end
            step=0.1;
            count_part=0;

            azero_i=a(i);
            azero_i(azero_i==0)=1;
            %ed_i=ent_part_der(A,labels,C,pcs,i,j);
            der_i=2*((D(:,i)'*D)*a-D(:,i)'*x)+lambda1*a(i)/abs(azero_i);
                                                   
            %ent_sum_no_ij=ent_sum(pcs([1:i-1, i+1:end],:));
            while(count_part<5 && step > 0.00006) %check 
                count_part=count_part+1;

                a_i_prev=a(i);
                %pc_row_prev=pcs(i,:);
                %ed_ij_prev=ed_i;
                der_i_prev=der_i;                
                %sum_A_rows_i_prev=sum_A_rows(i);%%%
                %sum_A_classes_ic_prev=sum_A_classes(i,labels(j));%%%

                %=== jump -->
                a(i)=a(i)-step*der_i;

                %======== it could be the calculation of just modified A_ij =========== 
                %======== and other elemnts could be get from previouse.    ===========                                           
                %norm_fro_pow2=norm(D*A-X,'fro')^2;

                dif=(a(i)-a_i_prev)*D(:,i);
                dif_pow2=(a(i)-a_i_prev)^2*D_pow2(:,i);
                toAdd_elemnts=dif_pow2+2*dif.*norm_fro_elements_base;%+
                norm_fro_pow2=norm_fro_pow2_base+sum(toAdd_elemnts);                        
                dif_abs=abs(a(i))-abs(a_i_prev);
                a_abs=a_abs_base+dif_abs;

                %sum_A_classes(i,labels(j))=sum_A_classes(i,labels(j))+dif_abs;%%%
                %sum_A_rows(i)=sum_A_rows(i)+dif_abs;%%%                        
%                 if(sum_A_rows(i)==0)%%%
%                     sum_A_rows(i)=1;%%%
%                 end%%%
%                 pc_row=sum_A_classes(i,:)/sum_A_rows(i);%%%
                %pc_row=compute_pc_row(A,labels,C,i); %%%
                %pcs(i,:)=pc_row;%%%
                %funcval2=norm_fro_pow2+lambda1*sum(sum(abs(A)))+lambda2*(ent_sum_no_ij+ent(pc_row,1));%norm(D*A-X,'fro')^2+lambda1*sum(sum(abs(A)))+lambda2*(ent_sum_no_ij+ent(pc_row,1));%ent_sum(pcs);
                funcval2=norm_fro_pow2+lambda1*a_abs;
                if(funcval2<funcval1)
                    count_better=count_better+1;
                    %worse=0;
                    step=step*1.9; %===== important: rate of increasing step size for faster convergence =======
                    funcval1=funcval2;

                    norm_fro_pow2_base=norm_fro_pow2;
                    norm_fro_elements_base=norm_fro_elements_base+dif;%toAdd_elemnts;
                    a_abs_base=a_abs;
                else
                    %worse=1;
                    step=step*0.4; %===== important: rate of decreasing step size for convergence, if we exceed the local min =======
                    a(i)=a_i_prev;                                        
                    der_i=der_i_prev;
                    %der_i_size=der_ij_size_prev;
                    %sum_A_rows(i)=sum_A_rows_i_prev;%%%
                    %sum_A_classes(i,labels(j))=sum_A_classes_ic_prev;%%%
                end                    
            end
            %funcval1_min_process(i,j)=funcval1;  
        end                        
        if(show_output)
            disp(strcat('target func value (afetr whole A update) = ',num2str(funcval1)));
            toc
        end
    end

end
end

