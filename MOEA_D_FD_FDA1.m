clear all;
clc;
tic;
%%definitions of dynamic problems
%%
N=100;  %% population size 
Gen=800; %% number of total generations
tao=10; % frequency of change
nt=10;  % severity of change
d=10;  %% dimensions of variables
Lb1=0;  %% lower range of x1
Ub1=1;   %% upper range of x1
Lb2=-1; % lower range of other variables
Ub2=1;  % upper range of other variables
p=0.8;  %  probability that parent solutions are selected from the neighborhood           
F=0.5;   %% scaled factor
CR=0.5;  %% corssover rate 
T=20; % neighborhood size
for i=1:Gen/tao     
    G(i)=sin(0.5*pi*(i-1)/nt);   % parameter of FDA1
end
%% initial weights
weight=zeros(N,2); % two-objectives problems
for i=1:N
    weight(i,1)=(i-1)/N;
    weight(i,2)=(N+1-i)/N;
end
for i=1:N
    for j=i+1:N
        Distance(i,j)=norm(weight(i,:)-weight(j,:),2); % compute E-distance of each pair of weights
        Distance(j,i)=Distance(i,j);
    end
    [~,index1]=sort(Distance(i,:));
    neighbor(i,:)=index1(1:N);    % neighbors of each weight
end
%%
 for i=1:N
     sol(i,1)=Lb1+(Ub1-Lb1)*rand;     %% initialize individuals 
     sol(i,2:d)=Lb2+(Ub2-Lb2)*rand(1,d-1);
     fitness(i,:)=fda1(sol(i,:),G(1));  % evaluate
 end
z(1)=min(fitness(:,1)); % initial ideal points
z(2)=min(fitness(:,2));
%%
for t=1:Gen
    K=ceil(t/tao);
         if t>tao && mod(t,tao)==1
           Fit(1:N,1:2,K-1)=fitness;   % store the previous POF
           all_s(:,:,K-1)=sol;         % store the previous POS
         
         if t>tao*2 && mod(t,tao)==1
            C1=mean(all_s(:,:,K-1),1);  % centroid of time K-1
            C2=mean(all_s(:,:,K-2),1);  % centroid of time K-2
             D1=C1-C2;  % their difference 
             for i=1:N
                 if mod(i,2)==0    % predict every two solutions
                     sol(i,:)=sol(i,:)+D1;  % the predicted solution
                     sol(i,1)=max(min(sol(i,1),Ub1),Lb1);
                     index22=find(sol(i,2:d)>Ub2); %repair each element if it is out of bounds
                     sol(i,index22+1)=Ub2;
                     index33=find(sol(i,2:d)<Lb2);
                     sol(i,index33+1)=Lb2;
                 end
            end
         end     
   
%{                        
         for i=1:N           %% this is another strategy to address DMOPs in which 20% of population are randomly reinitialized
             if mod(i,5)==0         
                 sol(i,1)=Lb1+(Ub1-Lb1)*rand;
                 sol(i,2:d)=Lb2+(Ub2-Lb2)*rand(1,d-1);
             end
         end
  %}       
     
          
            for i=1:N
               fitness(i,:)=fda1(sol(i,:),G(K));  % re-evaluate the new population
            end
          z(1)=min(fitness(:,1)); % initial ideal points
          z(2)=min(fitness(:,2));
          end
   
    for i=1:N
        P=[];
        if rand<p          % selection of mating
            P=neighbor(i,1:T);
        else P=[1:N];
        end
     
         No=length(P); % size of P
        dx=randperm(No);       
        w=dx(1);
        u=dx(2);  % randomly choose three indexs from P
        y=dx(3);
        for j=1:d % reproduction in each dimension
            if rand<CR
                V(j)=sol(P(y),j)+F*(sol(P(w),j)-sol(P(u),j));   %% generate a new individual            
            else V(j)=sol(i,j);
            end 
            if rand<0.5
               delta=(2*rand)^(1/21)-1;
            else delta=1-(2-2*rand)^(1/21);
            end
           if rand<(1/d)     % polynomial mutation
              if j==1
                   V(j)=V(j)+delta*(Ub1-Lb1);
              else V(j)=V(j)+delta*(Ub2-Lb2);
              end
           else V(j)=V(j);
           end
        end
           
        V(1)=max(min(V(1),Ub1),Lb1);
        index2=find(V(2:d)>Ub2); %repair each element if it is out of bounds
        V(index2+1)=Ub2;
        index3=find(V(2:d)<Lb2);
        V(index3+1)=Lb2;
        
        Fitness=fda1(V,G(K)); % evaluate this new individual
        
        if Fitness(1)<z(1) % update z
            z(1)=Fitness(1);
        end
        if Fitness(2)<z(2)
            z(2)=Fitness(2);
        end
  
       for j=1:No          % update the mating pool
          gg5=decom(sol(P(j),:),G(K),weight(P(j),:),z);
          gg6=decom(V,G(K),weight(P(j),:),z);
          if gg6<gg5
              sol(P(j),:)=V;
              fitness(P(j),:)=Fitness;
          end
       end 
    end  
%      plot(fitness(:,1),fitness(:,2),'o');
%      drawnow;
end
 Fit(:,:,K)=fitness;
 toc;
 %% IGD in each environment
 x=zeros(500,d);
for tt=1:Gen/tao 
for i=1:500
    x(i,1)=(i-1)/499;
    x(i,2:d)=repmat(G(1),1,d-1);
    tf(i,:)=fda1(x(i,:),G(1));
   for j=1:N 
    D(i,j)=norm(tf(i,:)-Fit(j,:,tt),2);
   end
    min_D(i)=min(D(i,:));
end
IGD(tt)=sum(min_D)/500;
IGD=IGD';
end
MIGD=mean(IGD);

%bar(IGD,0.4);  
    %% decomposition    
function g=decom(x,G,lmta,idp)
         fit=fda1(x,G);
         f(1)=lmta(1)*abs(fit(1)-idp(1));
        f(2)=lmta(2)*abs(fit(2)-idp(2));
         g=max(f);
end
%%
function f=fda1(x,G)
f(1)=x(1);
d=length(x);
s=0;
for i=2:d
    s=s+(x(i)-G)^2;
end
g=1+s;
h=1-sqrt(x(1)/g);
f(2)=g*h;
end
 