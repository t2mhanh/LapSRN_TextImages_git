
% -------------------------------------------------------------------------
%   Description:
%       Evaluate with Bicubic interpolation

%% testing options
dataset     = 'Test';
test_scale  = 4; %model_scale;  % testing scale can be different from model scale
compute_ifc = 0;            % IFC calculation is slow, enable when needed

%% setup paths
input_dir = fullfile('datasets',dataset);
output_dir = fullfile('results', dataset, sprintf('bicubicx%d', test_scale));

if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end



%% load image list
list_filename = fullfile('lists', sprintf('%s.txt', dataset));
% list_filename = fullfile('datasets', sprintf('%s.txt', dataset));
img_list = load_list(list_filename);
num_img = length(img_list);


%% testing
PSNR = zeros(num_img, 1);
SSIM = zeros(num_img, 1);
IFC  = zeros(num_img, 1);

for i = 1:num_img
    
    img_name = img_list{i};
    fprintf('Testing LapSRN on %s %dx: %d/%d: %s\n', dataset, test_scale, i, num_img, img_name);
    
    %% Load GT image
    input_filename = fullfile(input_dir, sprintf('%s.png', img_name));
    img_GT = im2double(imread(input_filename));
    img_GT = mod_crop(img_GT, test_scale);

    %% generate LR image
    img_LR = imresize(img_GT, 1/test_scale, 'bicubic');
    
    %% apply Bicubic
    img_HR = imresize(img_LR, test_scale, 'bicubic'); 
    
    %% save result
    output_filename = fullfile(output_dir, sprintf('%s.png', img_name));
    fprintf('Save %s\n', output_filename);
    imwrite(img_HR, output_filename);
        
    %% evaluate
    [PSNR(i), SSIM(i), IFC(i)] = evaluate_SR(img_GT, img_HR, test_scale, compute_ifc);
    
end

PSNR(end+1) = mean(PSNR);
SSIM(end+1) = mean(SSIM);
IFC(end+1)  = mean(IFC);

fprintf('Average PSNR = %f\n', PSNR(end));
fprintf('Average SSIM = %f\n', SSIM(end));
fprintf('Average IFC = %f\n', IFC(end));

filename = fullfile(output_dir, 'PSNR.txt');
save_matrix(PSNR, filename);

filename = fullfile(output_dir, 'SSIM.txt');
save_matrix(SSIM, filename);

filename = fullfile(output_dir, 'IFC.txt');
save_matrix(IFC, filename);

