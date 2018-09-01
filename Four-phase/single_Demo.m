
% Adaptive local fitting based active contour model for medical image segmentation
% Create Date: 1/8/2018
% Author: Dongdong Ma, Department of Eletronic Engineering, Tsinghua University

clc;
close all;
clear;

prefix = 'C:\Users\lenovo\Desktop\github\two-phase\data\';
img_num = 25;

img_name = strcat(prefix,'outImg',num2str(img_num),'.bmp');
Img = double(imread(img_name));
if size(Img,3) > 1
    Img = double(rgb2gray(uint8(Img)));
end

switch img_num
     case 17
        A = 255;
        timestep = 0.1;
        iter_outer = 80;                                  % inner iteration number
        iter_inner = 1;                                   % outer iteration number

        lamda1 = 1;
        lamda2 = 1.5;
        sigma = 5;                                        % control the size of neighboring region
        nu = 0.01 * A^2;                                  % coefficient of arc length term
        mu = 1;                                           % coefficient of distance regularization term
        c0 = 5;
        epsilon = 1;        
    case 19
        A = 255;
        timestep = 0.1;
        iter_outer = 80;                                  % outer iteration number
        iter_inner = 1;                                   % inner iteration number

        lamda1 = 1;
        lamda2 = 1.5;
        sigma = 5;                                        % control the size of neighboring region
        nu = 0.01 * A^2;                                  % coefficient of arc length term
        mu = 1;                                           % coefficient of distance regularization term
        c0 = 5;
        epsilon = 1;     
    case 21
        A = 255;
        timestep = 0.1;
        iter_outer = 80;                                  % outer iteration number
        iter_inner = 1;                                   % inner iteration number

        lamda1 = 1;
        lamda2 = 1.5;
        sigma = 5;                                        % control the size of neighboring region
        nu = 0.01 * A^2;                                  % coefficient of arc length term
        mu = 1;                                           % coefficient of distance regularization term
        c0 = 5;
        epsilon = 1;     
     case 23
        A = 255;
        timestep = 0.1;
        iter_outer = 80;                                  % outer iteration number
        iter_inner = 1;                                   % inner iteration number

        lamda1 = 1;
        lamda2 = 1.5;
        sigma = 5;                                        % control the size of neighboring region
        nu = 0.01 * A^2;                                  % coefficient of arc length term
        mu = 1;                                           % coefficient of distance regularization term
        c0 = 5;
        epsilon = 1;     
    case 25
        A = 255;
        timestep = 0.1;
        iter_outer = 80;                                  % outer iteration number
        iter_inner = 1;                                   % inner iteration number

        lamda1 = 1;
        lamda2 = 1.5;
        sigma = 5;                                        % control the size of neighboring region
        nu = 0.01 * A^2;                                  % coefficient of arc length term
        mu = 1;                                           % coefficient of distance regularization term
        c0 = 5;
        epsilon = 1;         
end

% initial contour
initialLSF = c0 * ones(size(Img));
initialLSF(50:end-50,50:end-50) = -c0;
u = initialLSF;

figure(2);
imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal
hold on;
contour(u,[0 0],'r');
title('Initial contour');

Komiga = ones(round(2*sigma)*2+1); Komiga = Komiga/sum(Komiga(:));
lamda_img = ones(size(Img));

rect_sdlen = (size(Komiga)-1)/2;
ImgExt = padarray(Img,rect_sdlen,'symmetric','both');
Imgblks = im2col(ImgExt,size(Komiga),'sliding');
Avg_Imgblks = mean(Imgblks);
sigma_mat = abs(Imgblks - repmat(Avg_Imgblks,size(Komiga,1)^2,1));
sigma2_mat = sigma_mat.^2;
I_sigma_mat = Imgblks.*sigma_mat;
omiga_mat = repmat(Komiga(:),1,size(Imgblks,2));

energy_curve = zeros(1,iter_outer);
for n = 1:iter_outer
    [u, lamda_img,all_pixel_energy] = alf_evolution(lamda1,lamda2,omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_inner);
    energy_curve(n) = all_pixel_energy;
    if mod(n,1) == 0
        pause(0.001);
        imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal;
        hold on;
        contour(u,[0 0],'r','LineWidth',1);
        iterNum = [num2str(n), ' iterations'];
        title(iterNum);
        hold off;
    end
end

% figure(5); plot(energy_curve);

% read the ground truth
gtruth_path = 'C:\Users\lenovo\Desktop\github\two-phase\data\';
file_name = strcat('outImg',num2str(img_num*5),'_crisp.bmp');
truth_bin = imread(strcat(gtruth_path,file_name));
whtmt = truth_bin == 3;                                % 2 means the white matter
mask = (truth_bin == 2 | truth_bin == 3 | truth_bin == 1 | truth_bin == 8);
output_img = u < 0;
output_img = output_img & mask;
figure(3); imshow(output_img);
figure(4); imshow(whtmt);title('ground truth');

TP_rate = TP_calculation(output_img,whtmt)
FP_rate = FP_calculation(output_img,whtmt)
P_rate = P_calculation(output_img,whtmt)
JCS_rate = JCS_calculation(output_img,whtmt)
DC_rate = DC_calculation(output_img,whtmt)

