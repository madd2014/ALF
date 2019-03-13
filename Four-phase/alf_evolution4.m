% evolution function

function [u, lamda_img,miu1_img,miu2_img,miu3_img,miu4_img] = alf_evolution4(lamda,omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u0,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_lse)

u = u0;

for kk = 1:iter_lse
    u(:,:,1) = NeumannBoundCond(u(:,:,1));
    K1 = curvature_central(u(:,:,1));                                              % div()
    DiracU1 = Dirac(u(:,:,1),epsilon);
    
    u(:,:,2) = NeumannBoundCond(u(:,:,2));
    K2 = curvature_central(u(:,:,2));                                              % div()
    DiracU2 = Dirac(u(:,:,2),epsilon);
    
    Hu1 = Heaviside(u(:,:,1),epsilon);
    Hu2 = Heaviside(u(:,:,2),epsilon);
    M1 = Hu1.*Hu2;          
    M2 = Hu2.*(1-Hu1);
    M3 = (1-Hu2).*Hu1; 
    M4 = (1 - Hu1).*(1 - Hu2); 

    M1Ext = padarray(M1,rect_sdlen,'symmetric','both');
    M1_blks = im2col(M1Ext,size(Komiga),'sliding');              
    M2Ext = padarray(M2,rect_sdlen,'symmetric','both');
    M2_blks = im2col(M2Ext,size(Komiga),'sliding');                 
    M3Ext = padarray(M3,rect_sdlen,'symmetric','both');
    M3_blks = im2col(M3Ext,size(Komiga),'sliding');                
    M4Ext = padarray(M4,rect_sdlen,'symmetric','both');
    M4_blks = im2col(M4Ext,size(Komiga),'sliding');                 

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

    temp = padarray(M4.*Img,rect_sdlen,'symmetric','both');
    temp1 = conv2(temp,Komiga,'valid');
    temp444 = sum(omiga_mat.*sigma_mat.*M4_blks);
    temp2 = lamda_img.*reshape(temp444,size(Img));
    temp = padarray(M4,rect_sdlen,'symmetric','both');
    temp3 = conv2(temp,Komiga,'valid');
    miu4_img = (temp1 - temp2)./(temp3 + exp(-99));

    % calculation of gradient flow
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

    temp = padarray(ones(size(Img)),rect_sdlen,'symmetric','both');
    temp1 = Img.^2.*conv2(temp,Komiga,'valid');
    temp = padarray(miu4_img.^2,rect_sdlen,'symmetric','both');
    temp2 = conv2(temp,Komiga,'valid');
    temp = padarray(lamda_img.^2,rect_sdlen,'symmetric','both');
    temp3 = diff_Img2.*conv2(temp,Komiga,'valid');
    temp = padarray(miu4_img,rect_sdlen,'symmetric','both');
    temp4 = 2*Img.*conv2(temp,Komiga,'valid');
    temp = padarray(lamda_img,rect_sdlen,'symmetric','both');
    temp5 = 2*Img.*diff_Img.*conv2(temp,Komiga,'valid');
    temp = padarray(lamda_img.*miu4_img,rect_sdlen,'symmetric','both');
    temp6 = 2*diff_Img.*conv2(temp,Komiga,'valid');
    e4x = temp1 + temp2 + temp3 - temp4 - temp5 + temp6;

    % multiply the weights
    e1x = lamda(1)*e1x;
    e2x = lamda(2)*e2x;
    e3x = lamda(3)*e3x;
    e4x = lamda(4)*e4x;
    
    dataForce1 = (e1x - e2x - e3x + e4x).*Hu2 + e3x - e4x;   
    ImageTerm = -DiracU1.*dataForce1;
    penalizeTerm = mu*(4*del2(u(:,:,1)) - K1);
    lengthTerm = nu.*DiracU1.*K1;
    u(:,:,1) = u(:,:,1) + timestep*(lengthTerm + penalizeTerm + ImageTerm);
    
    dataForce2 = (e1x - e2x - e3x + e4x).*Hu1 + e2x - e4x;    
    ImageTerm = -DiracU2.*dataForce2;
    penalizeTerm = mu*(4*del2(u(:,:,2)) - K2);
    lengthTerm = nu.*DiracU2.*K2;
    u(:,:,2) = u(:,:,2) + timestep*(lengthTerm + penalizeTerm + ImageTerm);
end

% update lamda
temp1 = sum(omiga_mat.*I_sigma_mat.*M1_blks);
temp1 = reshape(temp1,size(Img));
temp2 = sum(omiga_mat.*I_sigma_mat.*M2_blks);
temp2 = reshape(temp2,size(Img));
temp33 = sum(omiga_mat.*I_sigma_mat.*M3_blks);
temp33 = reshape(temp33,size(Img));
temp33_2 = sum(omiga_mat.*I_sigma_mat.*M4_blks);
temp33_2 = reshape(temp33_2,size(Img));

temp3 = miu1_img.*reshape(temp111,size(Img));
temp4 = miu2_img.*reshape(temp222,size(Img));
temp55 = miu3_img.*reshape(temp333,size(Img));
temp55_2 = miu4_img.*reshape(temp444,size(Img));

temp5 = sum(omiga_mat.*sigma2_mat.*M1_blks);
temp5 = reshape(temp5,size(Img));
temp6 = sum(omiga_mat.*sigma2_mat.*M2_blks);
temp6 = reshape(temp6,size(Img));
temp7 = sum(omiga_mat.*sigma2_mat.*M3_blks);
temp7 = reshape(temp7,size(Img));
temp7_2 = sum(omiga_mat.*sigma2_mat.*M4_blks);
temp7_2 = reshape(temp7_2,size(Img));

lamda_img = (temp1 - temp3 + temp2 - temp4 + temp33 - temp55 + temp33_2 - temp55_2)./(temp5 + temp6 + temp7 + temp7_2 + exp(-99));

test_end = 1;


% Neumann Bound condition
function g = NeumannBoundCond(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                       % the norm of the gradient plus a small possitive number                                                         % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,~] = gradient(Nx);                              
[~,nyy] = gradient(Ny);                              
k = nxx+nyy;                                            % compute divergence

function h = Heaviside(x,epsilon)    
h=0.5*(1+(2/pi)*atan(x./epsilon));

function f = Dirac(x, epsilon)    
f=(epsilon/pi)./(epsilon^2.+x.^2);



