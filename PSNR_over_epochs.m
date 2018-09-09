% compare PSNR, SSIM, IFC over number of epochs
% this file is used for x4 only. Modify for x2, x8.
% Run evaluation_over_epochs_TextImages.m for generating performance over training epochs
L1_GDL_model = './results/Test30/x4/LapSRN_x4/PSNR_SSIM_GDL_x4.mat'; 
load(L1_GDL_model)
PSNR_gdl = PSNR_epochs;
SSIM_gdl = SSIM_epochs;
IFC_gdl = IFC_epochs;

L1_model = './results/Test30/x4/LapSRN_x4/PSNR_SSIM_x4.mat';
load(L1_model)

%% PSNR
fig1 = figure(1);
num_epochs = 1000;
plot(1:num_epochs,PSNR_gdl(1:num_epochs),'-r',1:num_epochs,PSNR_epochs(1:num_epochs),'-g','LineWidth',2)
set(gcf,'color','w')
xlabel('Epoch')
ylabel('PSNR')
legend('L1-GDL','L1','Location','southeast')
print(fig1,['PSNR' num2str(num_epochs)],'-dpng')


fig2 = figure(2);
num_epochs = 200;
plot(1:num_epochs,PSNR_gdl(1:num_epochs),'-r',1:num_epochs,PSNR_epochs(1:num_epochs),'-g','LineWidth',2)
ylim([18 21])
xlabel('Epoch')
ylabel('PSNR')
legend('L1-GDL','L1','Location','southeast')
set(gcf,'color','w')
print(fig1,['PSNR' num2str(num_epochs)],'-dpng')

%% SSIM
close all
fig1 = figure(1);
num_epochs = 1000;
plot(1:num_epochs,SSIM_gdl(1:num_epochs),'-r',1:num_epochs,SSIM_epochs(1:num_epochs),'-g','LineWidth',2)
set(gcf,'color','w')
xlabel('Epoch')
ylabel('SSIM')
legend('L1-GDL','L1','Location','southeast')
print(fig1,['SSIM' num2str(num_epochs)],'-dpng')


fig2 = figure(2);
num_epochs = 200;
plot(1:num_epochs,SSIM_gdl(1:num_epochs),'-r',1:num_epochs,SSIM_epochs(1:num_epochs),'-g','LineWidth',2)
ylim([0.5 0.9])
xlabel('Epoch')
ylabel('SSIM')
legend('L1-GDL','L1','Location','southeast')
set(gcf,'color','w')
print(fig2,['SSIM' num2str(num_epochs)],'-dpng')

%% IFC
close all
fig1 = figure(1);
num_epochs = 1000;
plot(1:num_epochs,IFC_gdl(1:num_epochs),'-r',1:num_epochs,IFC_epochs(1:num_epochs),'-g','LineWidth',2)
set(gcf,'color','w')
xlabel('Epoch')
ylabel('IFC')
legend('L1-GDL','L1','Location','southeast')
print(fig1,['IFC' num2str(num_epochs)],'-dpng')


fig2 = figure(2);
num_epochs = 200;
plot(1:num_epochs,IFC_gdl(1:num_epochs),'-r',1:num_epochs,IFC_epochs(1:num_epochs),'-g','LineWidth',2)
ylim([1.5 2.1])
xlabel('Epoch')
ylabel('IFC')
set(gcf,'color','w')
legend('L1-GDL','L1','Location','southeast')
print(fig2,['IFC' num2str(num_epochs)],'-dpng')

