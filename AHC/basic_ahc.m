lambdaw = 0.9;
lambdav = 0.8;
alpha = 1000;
beta = 0.5;
gamma =0.95;

w=zeros(162,1);
v=zeros(162,1);
x_bar=zeros(162,1);
e=zeros(162,1);

x_encoder=zeros(162,1);               %creat a zero matrix
x=0;
x_dot=0;
theta=0;
theta_dot=0;

%%%%%%%%%%%%%%%%%%%%%ACE
x_bar = x_bar+(1-lambdav)*x_encoder;  %Update ACE eligibility:
p=sum(v.*x_encoder);                  %Compute p
r_hat = reward + gamma*p-pre_p;       % Compute r_hat
v=v+beta*r_hat*pre_x_bar;             %Update ACE weights v

%%%%%%%%%%%%%%%%%%%%%ASE
w=w+alpha*r_hat.*pre_e;              %Update ASE weights w
e=(1-lambdaw)*y*x_encoder;           %Update ASE eligibility

%%%%%%%%%%%%%%%%%%%%decide action
y=sum(w.*x_encoder);

pre_x_bar=x_bar;
pre_p=p;
pre_e=e;



