
clc;
close all;
clear;                        

addpath('F:\signal_processing20190112审稿意见\增补实验\append_exp7\14_MICO_v0\MICO_3D');
data = load('F:\signal_processing20190112审稿意见\增补实验\append_exp7\Img_3D.mat');
Img = data.Img_3D;

th_bg = 5;
[x1, x2, y1, y2, z1, z2] = cropImg(Img, th_bg);        % crop image
[DimX1, DimY1, DimZ1] = size(Img);
x1 = max(1,x1-2); x2 = min(DimX1,x2+2);
y1 = max(1,y1-2); y2 = min(DimY1,y2+2);
z1 = max(1,z1-2); z2 = min(DimZ1,z2+2);
Img3D = Img(x1:x2,y1:y2,z1:z2);
[DimX, DimY, DimZ] = size(Img3D);   

c0 = 5;
M = zeros(DimX, DimY, DimZ);
M(round(DimX/2) - 30:round(DimX/2) + 30, round(DimY/2) - 30:round(DimY/2) + 30, round(DimZ/2) - 30:round(DimZ/2) + 30) = c0;
u = M;

timestep = 0.1;
iter_outer = 100;                          
iter_inner = 1;                                
lamda1 = 1;
lamda2 = 0.65;
sigma = 1;                                    
nu = 0.0005 * 255^2;                             
mu = 1;                                           
epsilon = 1;        


K = fspecial('gaussian',round(2*sigma)*2+1,sigma);     % the Gaussian kernel
Komiga_3D = zeros(round(2*sigma)*2+1,round(2*sigma)*2+1,round(2*sigma)*2+1);
for ii = 1:size(Komiga_3D,3)
%     Ksigma_3D(:,:,ii) = K;
    Komiga_3D(:,:,ii) = ones(round(2*sigma)*2+1);
end
Komiga_3D = Komiga_3D/sum(Komiga_3D(:));
lamda_img_3D = ones(size(Img3D));


mean_cube_kernel = Komiga_3D;
Avg_Imgblks = convn(Img3D,mean_cube_kernel,'same');

sigma_mat = abs(Img3D - Avg_Imgblks);
sigma2_mat = sigma_mat.^2;
I_sigma_mat = Img3D.*sigma_mat;

tic;
for n = 1:iter_outer
    [u, lamda_img_3D] = last_lse_bfe_3D(lamda1,lamda2,sigma_mat,sigma2_mat,I_sigma_mat,u,Img3D,lamda_img_3D,Komiga_3D,nu,timestep,mu,epsilon,iter_inner);
    if mod(n,1) == 0
        pause(0.001);
        imagesc(Img3D(:,:,100),[0, 255]); colormap(gray); axis off; axis equal;
        hold on;
        contour(u(:,:,100),[0 0],'r','LineWidth',1);
        iterNum = [num2str(n), ' iterations'];
        title(iterNum);
        hold off;
    end
end
toc;

save proposed_result_3D.mat u

gtruth_path = 'F:\level set（5.08）\当前仿真代码\18_My_ALG\18_my_local_adpt_improve_2\exp\data\';
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


test_end = 1;


