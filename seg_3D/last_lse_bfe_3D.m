
function [u, lamda_img_3D] = last_lse_bfe_3D(lamda1,lamda2,sigma_mat,sigma2_mat,I_sigma_mat,u0,Img3D,lamda_img_3D,Komiga_3D,nu,timestep,mu,epsilon,iter_lse)

u = u0;
Hu = Heaviside(u,epsilon);
M1 = Hu;                                                   % M(:,:,1) = Hu;
M2 = 1-Hu;                                                 % M(:,:,2) = 1-Hu;

temp1 = convn(M1.*Img3D,Komiga_3D,'same');
temp2 = lamda_img_3D.*convn(M1.*sigma_mat,Komiga_3D,'same');
temp3 = convn(M1,Komiga_3D,'same');
miu1_img = (temp1 - temp2)./(temp3 + exp(-99));

temp1 = convn(M2.*Img3D,Komiga_3D,'same');
temp2 = lamda_img_3D.*convn(M2.*sigma_mat,Komiga_3D,'same');
temp3 = convn(M2,Komiga_3D,'same');
miu2_img = (temp1 - temp2)./(temp3 + exp(-99));

% 自己重新编写的求三维梯度下降流的代码
temp1 = Img3D.^2.*convn(ones(size(Img3D)),Komiga_3D,'same');
temp2 = convn(miu1_img.^2,Komiga_3D,'same');
temp3 = sigma2_mat.*convn(lamda_img_3D.^2,Komiga_3D,'same');
temp4 = 2*Img3D.*convn(miu1_img,Komiga_3D,'same');
temp5 = 2*Img3D.*sigma_mat.*convn(lamda_img_3D,Komiga_3D,'same');
temp6 = 2*sigma_mat.*convn(miu1_img.*lamda_img_3D,Komiga_3D,'same');
e1x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

temp1 = Img3D.^2.*convn(ones(size(Img3D)),Komiga_3D,'same');
temp2 = convn(miu2_img.^2,Komiga_3D,'same');
temp3 = sigma2_mat.*convn(lamda_img_3D.^2,Komiga_3D,'same');
temp4 = 2*Img3D.*convn(miu2_img,Komiga_3D,'same');
temp5 = 2*Img3D.*sigma_mat.*convn(lamda_img_3D,Komiga_3D,'same');
temp6 = 2*sigma_mat.*convn(miu2_img.*lamda_img_3D,Komiga_3D,'same');
e2x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

for kk = 1:iter_lse
    
    for ii = 1:size(u,3)
        temp = u(:,:,ii);
        u(:,:,ii) = NeumannBoundCond(temp);
    end
    
    K = curvature_central_3D(u);      
    DiracU = Dirac(u,epsilon);
    penalizeTerm = mu*(4*del2(u) - K);   
    lengthTerm = nu.*DiracU.*K;
    
    ImageTerm = -DiracU.*(lamda1*e1x - lamda2*e2x);


    u = u + timestep*(lengthTerm + penalizeTerm + ImageTerm);
end

temp1 = convn(I_sigma_mat.*M1,Komiga_3D,'same');
temp2 = miu1_img.*convn(sigma_mat.*M1,Komiga_3D,'same');
temp3 = convn(I_sigma_mat.*M2,Komiga_3D,'same');
temp4 = miu2_img.*convn(sigma_mat.*M2,Komiga_3D,'same');
temp5 = convn(sigma2_mat.*M1,Komiga_3D,'same');
temp6 = convn(sigma2_mat.*M2,Komiga_3D,'same');
lamda_img_3D = (temp1 - temp2 + temp3 - temp4)./(temp5 + temp6 + exp(-99));

test_end = 1;

function g = NeumannBoundCond(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)

[ux,uy] = gradient(u);
normDu = sqrt(ux.^2+uy.^2+1e-10);
Nx = ux./normDu;
Ny = uy./normDu;
[nxx,~] = gradient(Nx);
[~,nyy] = gradient(Ny);
k = nxx+nyy;

function k = curvature_central_3D(u)                       
% compute curvature
[ux,uy,uz] = gradient(u);                                  
normDu = sqrt(ux.^2 + uy.^2 + uz.^2 + 1e-10);
Nx = ux./normDu;                                       
Ny = uy./normDu;
Nz = uz./normDu;
[nxx,~,~] = gradient(Nx);                              
[~,nyy,~] = gradient(Ny); 
[~,~,nzz] = gradient(Nz);    
k = nxx + nyy + nzz;                                             % compute divergence

function h = Heaviside(x,epsilon)    
h = 0.5*(1+(2/pi)*atan(x./epsilon));

function f = Dirac(x,epsilon)    
f = (epsilon/pi)./(epsilon^2.+x.^2);
% b = (x <= epsilon) & (x >= -epsilon);
% f = f.*b;



