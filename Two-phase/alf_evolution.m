% evolution function

function [u, lamda_img,all_pixel_energy]= alf_evolution(lamda1,lamda2,omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u0,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_lse)

u = u0;
Hu = Heaviside(u,epsilon);
M1 = Hu;                                                   % M(:,:,1) = Hu;
M2 = 1-Hu;                                                 % M(:,:,2) = 1-Hu;
M1Ext = padarray(M1,rect_sdlen,'symmetric','both');
M1_blks = im2col(M1Ext,size(Komiga),'sliding');             
M2Ext = padarray(M2,rect_sdlen,'symmetric','both');
M2_blks = im2col(M2Ext,size(Komiga),'sliding');                    

temp = padarray(M1.*Img,rect_sdlen,'symmetric','both');
temp1 = conv2(temp,Komiga,'valid');
temp111 = sum(omiga_mat.*sigma_mat.*M1_blks);
temp2 = lamda_img.*reshape(temp111,size(Img));
temp = padarray(M1,rect_sdlen,'symmetric','both');
temp3 = conv2(temp,Komiga,'valid');
miu1_img = (temp1 - temp2)./(temp3 + + exp(-99));

temp = padarray(M2.*Img,rect_sdlen,'symmetric','both');
temp1 = conv2(temp,Komiga,'valid');
temp222 = sum(omiga_mat.*sigma_mat.*M2_blks);
temp2 = lamda_img.*reshape(temp222,size(Img));
temp = padarray(M2,rect_sdlen,'symmetric','both');
temp3 = conv2(temp,Komiga,'valid');
miu2_img = (temp1 - temp2)./(temp3 + exp(-99));

% calculate the gradient flow
temp = sigma_mat((size(Komiga,1)^2+1)/2,:);       
diff_Img = reshape(temp,size(Img));
diff_Img2 = diff_Img.^2;
temp = padarray(ones(size(Img)),rect_sdlen,'symmetric','both');
temp1 = Img.^2.*conv2(temp,Komiga,'valid');
temp = padarray(miu1_img.^2,rect_sdlen,'symmetric','both');
temp2 = conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.^2,rect_sdlen,'symmetric','both');
temp3 = diff_Img2.*conv2(temp,Komiga,'valid');
temp = padarray(miu1_img,rect_sdlen,'symmetric','both');
temp4 = 2*Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img,rect_sdlen,'symmetric','both');
temp5 = 2*Img.*diff_Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.*miu1_img,rect_sdlen,'symmetric','both');
temp6 = 2*diff_Img.*conv2(temp,Komiga,'valid');
e1x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

temp = padarray(ones(size(Img)),rect_sdlen,'symmetric','both');
temp1 = Img.^2.*conv2(temp,Komiga,'valid');
temp = padarray(miu2_img.^2,rect_sdlen,'symmetric','both');
temp2 = conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.^2,rect_sdlen,'symmetric','both');
temp3 = diff_Img2.*conv2(temp,Komiga,'valid');
temp = padarray(miu2_img,rect_sdlen,'symmetric','both');
temp4 = 2*Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img,rect_sdlen,'symmetric','both');
temp5 = 2*Img.*diff_Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.*miu2_img,rect_sdlen,'symmetric','both');
temp6 = 2*diff_Img.*conv2(temp,Komiga,'valid');
e2x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

% calculate the whole energy
temp = M1.*e1x/lamda1 + M2.*e2x/lamda2;
all_pixel_energy = sum(temp(:));

for kk = 1:iter_lse
    u = NeumannBoundCond(u);
    K = curvature_central(u);                                              % div()
    DiracU = Dirac(u,epsilon);
    ImageTerm = -DiracU.*(lamda1*e1x - lamda2*e2x);
    penalizeTerm = mu*(4*del2(u) - K);  
    lengthTerm = nu.*DiracU.*K;
    u = u + timestep*(lengthTerm + penalizeTerm + ImageTerm);
end

% 下面是更新lamda矩阵
temp1 = sum(omiga_mat.*I_sigma_mat.*M1_blks);
temp1 = reshape(temp1,size(Img));
temp2 = sum(omiga_mat.*I_sigma_mat.*M2_blks);
temp2 = reshape(temp2,size(Img));
temp3 = miu1_img.*reshape(temp111,size(Img));
temp4 = miu2_img.*reshape(temp222,size(Img));
temp5 = sum(omiga_mat.*sigma2_mat.*M1_blks);
temp5 = reshape(temp5,size(Img));
temp6 = sum(omiga_mat.*sigma2_mat.*M2_blks);
temp6 = reshape(temp6,size(Img));
lamda_img = (temp1 - temp3 + temp2 - temp4)./(temp5 + temp6 + exp(-99));

test_end = 1;


% Neumann Bound condition
function g = NeumannBoundCond(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)
% central difference
[ux,uy] = gradient(u);
normDu = sqrt(ux.^2+uy.^2+1e-10);
Nx = ux./normDu;
Ny = uy./normDu;
[nxx,~] = gradient(Nx);
[~,nyy] = gradient(Ny);
k = nxx+nyy;

function h = Heaviside(x,epsilon)    
h=0.5*(1+(2/pi)*atan(x./epsilon));

function f = Dirac(x, epsilon)    
f=(epsilon/pi)./(epsilon^2.+x.^2);



