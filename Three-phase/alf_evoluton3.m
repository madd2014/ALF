% evolution function

function [u, lamda_img,miu1_img,miu2_img,miu3_img]= alf_evoluton3(omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u0,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_lse)

u = u0;
Hu1 = Heaviside(u(:,:,1),epsilon);
Hu2 = Heaviside(u(:,:,2),epsilon);
M1 = Hu1.*Hu2;          
M2 = Hu1.*(1-Hu2);
M3 = 1 - Hu1; 
M1Ext = padarray(M1,rect_sdlen,'symmetric','both');
M1_blks = im2col(M1Ext,size(Komiga),'sliding');         
M2Ext = padarray(M2,rect_sdlen,'symmetric','both');
M2_blks = im2col(M2Ext,size(Komiga),'sliding');         
M3Ext = padarray(M3,rect_sdlen,'symmetric','both');
M3_blks = im2col(M3Ext,size(Komiga),'sliding');       

% update miu
temp = padarray(M1.*Img,rect_sdlen,'symmetric','both');
temp1 = conv2(temp,Komiga,'valid');
temp111 = sum(omiga_mat.*sigma_mat.*M1_blks);
temp2 = lamda_img.*reshape(temp111,size(Img));
temp = padarray(M1,rect_sdlen,'symmetric','both');
temp3 = conv2(temp,Komiga,'valid');
miu1_img = (temp1 - temp2)./(temp3 + exp(-99));

temp = padarray(M2.*Img,rect_sdlen,'symmetric','both');
temp1 = conv2(temp,Komiga,'valid');
temp222 = sum(omiga_mat.*sigma_mat.*M2_blks);
temp2 = lamda_img.*reshape(temp222,size(Img));
temp = padarray(M2,rect_sdlen,'symmetric','both');
temp3 = conv2(temp,Komiga,'valid');
miu2_img = (temp1 - temp2)./(temp3 + exp(-99));

temp = padarray(M3.*Img,rect_sdlen,'symmetric','both');
temp1 = conv2(temp,Komiga,'valid');
temp333 = sum(omiga_mat.*sigma_mat.*M3_blks);
temp2 = lamda_img.*reshape(temp333,size(Img));
temp = padarray(M3,rect_sdlen,'symmetric','both');
temp3 = conv2(temp,Komiga,'valid');
miu3_img = (temp1 - temp2)./(temp3 + exp(-99));

% calculate the gradient flow
temp = sigma_mat((size(Komiga,1)^2+1)/2,:);                                 % 块中心与均值的差值，也就是sigma(x)
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

temp = padarray(ones(size(Img)),rect_sdlen,'symmetric','both');
temp1 = Img.^2.*conv2(temp,Komiga,'valid');
temp = padarray(miu3_img.^2,rect_sdlen,'symmetric','both');
temp2 = conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.^2,rect_sdlen,'symmetric','both');
temp3 = diff_Img2.*conv2(temp,Komiga,'valid');
temp = padarray(miu3_img,rect_sdlen,'symmetric','both');
temp4 = 2*Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img,rect_sdlen,'symmetric','both');
temp5 = 2*Img.*diff_Img.*conv2(temp,Komiga,'valid');
temp = padarray(lamda_img.*miu3_img,rect_sdlen,'symmetric','both');
temp6 = 2*diff_Img.*conv2(temp,Komiga,'valid');
e3x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

for kk = 1:iter_lse
    u(:,:,1) = NeumannBoundCond(u(:,:,1));
    K = curvature_central(u(:,:,1));                                              % div()
    DiracU = Dirac(u(:,:,1),epsilon);
    ImageTerm = -DiracU.*(e1x.*Hu2 + e2x.*(1-Hu2) - e3x);
    penalizeTerm = mu*(4*del2(u(:,:,1)) - K);                            
    lengthTerm = nu.*DiracU.*K;
    u(:,:,1) = u(:,:,1) + timestep*(lengthTerm + penalizeTerm + ImageTerm);
    
    u(:,:,2) = NeumannBoundCond(u(:,:,2));
    K = curvature_central(u(:,:,2));                                              % div()
    DiracU = Dirac(u(:,:,2),epsilon);
    ImageTerm = -DiracU.*(e1x - e2x).*Hu1;
    penalizeTerm = mu*(4*del2(u(:,:,2)) - K);
    lengthTerm = nu.*DiracU.*K;
    u(:,:,2) = u(:,:,2) + timestep*(lengthTerm + penalizeTerm + ImageTerm);
end

% update lamda
temp1 = sum(omiga_mat.*I_sigma_mat.*M1_blks);
temp1 = reshape(temp1,size(Img));
temp2 = sum(omiga_mat.*I_sigma_mat.*M2_blks);
temp2 = reshape(temp2,size(Img));
temp33 = sum(omiga_mat.*I_sigma_mat.*M3_blks);
temp33 = reshape(temp33,size(Img));
temp3 = miu1_img.*reshape(temp111,size(Img));
temp4 = miu2_img.*reshape(temp222,size(Img));
temp55 = miu3_img.*reshape(temp333,size(Img));
temp5 = sum(omiga_mat.*sigma2_mat.*M1_blks);
temp5 = reshape(temp5,size(Img));
temp6 = sum(omiga_mat.*sigma2_mat.*M2_blks);
temp6 = reshape(temp6,size(Img));
temp7 = sum(omiga_mat.*sigma2_mat.*M3_blks);
temp7 = reshape(temp7,size(Img));
lamda_img = (temp1 - temp3 + temp2 - temp4 + temp33 - temp55)./(temp5 + temp6 + temp7 + exp(-99));

% test_end = 1;


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



