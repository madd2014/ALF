% Adaptive local fitting based active contour model for medical image
% segmentation, three phase segmentation
% Create Date: 1/8/2018
% Author: Dongdong Ma, Department of Eletronic Engineering, Tsinghua University
clc;
close all;
clear;

name_set = {'1.bmp','6.bmp','34.bmp','35.png'};
prefix = 'C:\Users\lenovo\Desktop\github\three-phase\data\';
name_num = 1;
img_name = strcat(prefix,name_set{name_num});
Img = double(imread(img_name));
if size(Img,3) > 1
    Img = double(rgb2gray(uint8(Img)));
end
figure;
imagesc(Img,[0 255]);colormap(gray);hold on; axis off;axis equal;
title('Initial contour');

switch name_num
    case 1
        A = 255;
        timestep = 0.1;
        iter_outer = 200;                                  % outer iteration number
        iter_inner = 1;                                    % inner iteration number
        sigma = 5;                                         % control the size of neighboring region
        nu = 0.005 * A^2;                                  % coefficient of arc length term
        mu = 1.5;                                          % coefficient of distance regularization term
        epsilon = 1;
        
        c0 = 3;
        temp1 = c0 * ones(size(Img));
        temp1(30:end-15,30:end-30) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;     
    case 2
        A = 255;
        timestep = 0.1;
        iter_outer = 50;                                    % outer iteration number
        iter_inner = 1;                                     % inner iteration number
        sigma = 8;                                          % control the size of neighboring region
        nu = 0.005 * A^2;                                   % coefficient of arc length term
        mu = 1.5;                                           % coefficient of distance regularization term
        epsilon = 1;

        c0 = 3;
        temp1 = c0 * ones(size(Img));
        temp1(35:end-35,35:end-35) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
    case 3
        A = 255;
        timestep = 0.1;
        iter_outer = 200;                                  % outer iteration number
        iter_inner = 1;                                    % inner iteration number
        sigma = 5;                                         % control the size of neighboring region
        nu = 0.003 * A^2;                                  % coefficient of arc length term
        mu = 1.5;                                          % coefficient of distance regularization term
        epsilon = 1;

        c0 = 3;
        temp1 = c0 * ones(size(Img));
        temp1(40:end-25,40:end-40) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(42:end-23,42:end-38) = -c0; 
        initialLSF(:,:,2) = temp2;
    case 4

        A = 255;
        timestep = 0.1;
        iter_outer = 150;                                   % outer iteration number
        iter_inner = 1;                                     % inner iteration number
        sigma = 5;                                          % control the size of neighboring region
        nu = 0.005 * A^2;                                   % coefficient of arc length term
        mu = 1.5;                                           % coefficient of distance regularization term
        epsilon = 1;

        c0 = 8;
        temp1 = c0 * ones(size(Img));
        temp1(55:end-55,40:end-40) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(30:end-20,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
    case 5
        A = 255;
        timestep = 0.1;
        iter_outer = 150;                                   % outer iteration number
        iter_inner = 1;                                     % inner iteration number
        sigma = 5;                                          % control the size of neighboring region
        nu = 0.0001 * A^2;                                  % coefficient of arc length term
        mu = 1.5;                                           % coefficient of distance regularization term
        epsilon = 1;

        c0 = 3;
        temp1 = c0 * ones(size(Img));
        temp1(30:end-15,30:end-30) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
        
end

u = initialLSF;
[c,h] = contour(u(:,:,1),[0 0],'r','LineWidth',1);
[c,h] = contour(u(:,:,2),[0 0],'g','LineWidth',1);
hold off

Komiga = ones(round(2*sigma)*2 + 1); Komiga = Komiga/sum(Komiga(:));
lamda_img = ones(size(Img));

rect_sdlen = (size(Komiga)-1)/2;
ImgExt = padarray(Img,rect_sdlen,'symmetric','both');
Imgblks = im2col(ImgExt,size(Komiga),'sliding');
Avg_Imgblks = mean(Imgblks);
sigma_mat = abs(Imgblks - repmat(Avg_Imgblks,size(Komiga,1)^2,1));
sigma2_mat = sigma_mat.^2;
I_sigma_mat = Imgblks.*sigma_mat;
omiga_mat = repmat(Komiga(:),1,size(Imgblks,2));

for n = 1:iter_outer
    lamda_img = zeros(size(Img));
    [u,lamda_img,miu1_img,miu2_img,miu3_img] = alf_evoluton3(omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_inner);
    if mod(n,1) == 0
        pause(0.001);
        imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal;
        hold on;
        [c,h] = contour(u(:,:,1),[0 0],'r','LineWidth',1);
        [c,h] = contour(u(:,:,2),[0 0],'g','LineWidth',1);
        iterNum = [num2str(n), ' iterations'];
        title(iterNum);
        hold off;
    end
end

H1 =  Heaviside(u(:,:,1),epsilon );
H2 =  Heaviside(u(:,:,2),epsilon );
M1 = H1.*H2;
M2 = H1.*(1-H2);
M3 = (1-H1);
figure(2); imshow(M1);
figure(3); imshow(M2);
figure(4); imshow(M3);

Img_seg = mean(miu1_img(:))*M1 + mean(miu2_img(:))*M2 + mean(miu3_img(:))*M3;  % three regions are labeled with C1, C2, C3
figure; imagesc(Img_seg); axis off; axis equal; title('Segmented regions');
colormap(gray);


