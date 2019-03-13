% Adaptive local fitting based active contour model for medical image segmentation
% Create Date: 1/8/2018
% Author: Dongdong Ma, Department of Eletronic Engineering, Tsinghua University
clc;
close all;
clear;

prefix = 'C:\Users\lenovo\Desktop\github\four-phase\data\';
img_num = 22;
img_name = strcat(prefix,'outImg',num2str(img_num),'.bmp');
Img = double(imread(img_name));
if size(Img,3) > 1
    Img = double(rgb2gray(uint8(Img)));
end

lamda = ones(1,4);
switch img_num
    case 19
        A = 255;
        timestep = 0.1;
        iter_outer = 30;                                    % 外循环次数
        iter_inner = 1;                                     % 内循环次数
        sigma = 7;                                          % 邻域的大小
        nu = 0.0057 * A^2;                                  % 弧长项系数
        mu = 1.0;                                           % 距离规则项的系数
        epsilon = 1;
        
        lamda(1) = 1.2;
        lamda(2) = 2.0;
        lamda(3) = 1.3;
        lamda(4) = 2.0;
        
        % 7月17日，自己定义的初始化
        c0 = 3.0;
        temp1 = c0 * ones(size(Img));
        temp1(50:end-50,50:end-50) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
        
    case 20

        A = 255;
        timestep = 0.1;
        iter_outer = 30;                                  % 外循环次数
        iter_inner = 1;                                     % 内循环次数
        sigma = 7;                                          % 邻域的大小
        nu = 0.006 * A^2;                                  % 弧长项系数
        mu = 1.0;                                           % 距离规则项的系数
        epsilon = 1;
        
        lamda(1) = 1.8;
        lamda(2) = 3.2;
        lamda(3) = 1.9;
        lamda(4) = 3.2;
        
        c0 = 3.0;
        temp1 = c0 * ones(size(Img));
        temp1(50:end-50,50:end-50) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
    case 21
        A = 255;
        timestep = 0.1;
        iter_outer = 30;                                  % 外循环次数
        iter_inner = 1;                                     % 内循环次数
        sigma = 7;                                          % 邻域的大小
        nu = 0.005 * A^2;                                  % 弧长项系数
        mu = 1.0;                                           % 距离规则项的系数
        epsilon = 1;
        
        lamda(1) = 1.0;
        lamda(2) = 2.0;
        lamda(3) = 1.2;
        lamda(4) = 2.1;
       
        c0 = 5.0;
        temp1 = c0 * ones(size(Img));
        temp1(50:end-50,50:end-50) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;

    case 22
        A = 255;
        timestep = 0.1;
        iter_outer = 30;                                  % 外循环次数
        iter_inner = 1;                                     % 内循环次数
        sigma = 7;                                          % 邻域的大小
        nu = 0.0055 * A^2;                                  % 弧长项系数
        mu = 1.0;                                           % 距离规则项的系数
        epsilon = 1;

        lamda(1) = 1.0;
        lamda(2) = 2.0;
        lamda(3) = 1.2;
        lamda(4) = 2.1;        
        
        c0 = 5.0;
        temp1 = c0 * ones(size(Img));
        temp1(50:end-50,50:end-50) = -c0; 
        initialLSF(:,:,1) = temp1;
        temp2 = c0 * ones(size(Img));
        temp2(10:end-10,10:end-10) = -c0; 
        initialLSF(:,:,2) = temp2;
        
end

u = initialLSF;
figure(1);
imagesc(Img,[0 255]);colormap(gray);hold on; axis off;axis equal;
[c,h] = contour(u(:,:,1),[0 0],'r','LineWidth',1);
[c,h] = contour(u(:,:,2),[0 0],'g','LineWidth',1);

Komiga = ones(round(2*sigma)*2 + 1); Komiga = Komiga/sum(Komiga(:));
Komiga = ones(round(2*sigma)*2 + 1); Komiga = Komiga/sum(Komiga(:));
lamda_img = ones(size(Img));

% update miu
rect_sdlen = (size(Komiga)-1)/2;
ImgExt = padarray(Img,rect_sdlen,'symmetric','both');
Imgblks = im2col(ImgExt,size(Komiga),'sliding');
Avg_Imgblks = mean(Imgblks);
sigma_mat = abs(Imgblks - repmat(Avg_Imgblks,size(Komiga,1)^2,1));
sigma2_mat = sigma_mat.^2;
I_sigma_mat = Imgblks.*sigma_mat;
omiga_mat = repmat(Komiga(:),1,size(Imgblks,2));

for n = 1:iter_outer
    [u,lamda_img,miu1_img,miu2_img,miu3_img,miu4_img] = alf_evolution4(lamda,omiga_mat,sigma_mat,sigma2_mat,I_sigma_mat,rect_sdlen,u,Img,lamda_img,Komiga,nu,timestep,mu,epsilon,iter_inner);
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

H1 = 0.5*(1+(2/pi)*atan(u(:,:,1)./epsilon));  
H2 = 0.5*(1+(2/pi)*atan(u(:,:,2)./epsilon));
M1 = H1.*H2;
M2 = H2.*(1-H1);
M3 = H1.*(1-H2);
M4 = (1-H1).*(1-H2);

Img_seg = mean(miu1_img(:))*M1 + mean(miu2_img(:))*M2 + mean(miu3_img(:))*M3 + mean(miu4_img(:))*M4;  % three regions are labeled with C1, C2, C3
figure(2); imagesc(Img_seg); axis off; axis equal; title('Segmented regions');
colormap(gray);

% read ground truth
% 0: background;
% 1: CSF;
% 2: Grey Matter;
% 3: White Matter;
gtruth_path = 'C:\Users\lenovo\Desktop\github\four-phase\data\';
file_name = strcat('outImg',num2str(5*img_num),'_crisp.bmp');
truth_bin = imread(strcat(gtruth_path,file_name));
CSF = truth_bin == 1;
gry = truth_bin == 2;
whtmt = truth_bin == 3;                                % 2 means the white matter
mask = (truth_bin == 2 | truth_bin == 3 | truth_bin == 1 | truth_bin == 8);

output_img1 = M1 > 0.5;
output_img2 = M2 > 0.5;
output_img3 = M3 > 0.5;
output_img4 = M4 > 0.5;
output_img1 = output_img1 & mask;
output_img2 = output_img2 & mask;
output_img3 = output_img3 & mask;
output_img4 = output_img4 & mask;
figure(3); imshow(output_img1);
figure(4); imshow(output_img2);
figure(5); imshow(output_img3);
figure(6); imshow(output_img4);

figure(7); imshow(CSF);title('CSF ground truth');
figure(8); imshow(gry);title('gry ground truth');
figure(9); imshow(whtmt);title('whtmt ground truth');

% calculate index for CSF
CSF_TP_rate = TP_calculation(output_img1,CSF)
CSF_FP_rate = FP_calculation(output_img1,CSF)
CSF_P_rate = P_calculation(output_img1,CSF)
CSF_JCS_rate = JCS_calculation(output_img1,CSF)
CSF_DC_rate = DC_calculation(output_img1,CSF)
% calculate index for grey matter
gray_TP_rate = TP_calculation(output_img3,gry)
gray_FP_rate = FP_calculation(output_img3,gry)
gray_P_rate = P_calculation(output_img3,gry)
gray_JCS_rate = JCS_calculation(output_img3,gry)
gray_DC_rate = DC_calculation(output_img3,gry)
% calculate index for white matter
whtmt_TP_rate = TP_calculation(output_img4,whtmt)
whtmt_FP_rate = FP_calculation(output_img4,whtmt)
whtmt_P_rate = P_calculation(output_img4,whtmt)
whtmt_JCS_rate = JCS_calculation(output_img4,whtmt)
whtmt_DC_rate = DC_calculation(output_img4,whtmt)


