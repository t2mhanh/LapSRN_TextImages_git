function Y = L2GDL_loss(X, Z,gdl_lambda, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L2 (MSE) loss function used in MatConvNet NN
%       forward : Y = vllab_nn_L2_loss(X, Z)
%       backward: Y = vllab_nn_L2_loss(X, Z, dzdy)
%
%   Input:
%       - X     : predicted data
%       - Z     : ground truth data
%       - dzdy  : the derivative of the output
%
%   Output:
%       - Y     : loss when forward, derivative of loss when backward
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

% L = 1/2*(x-Z).^2 + (|Xi,j - Xi-1,j| - |Zi,j - Zi-1,j|).^2 + (|Xi,j - Xi,j-1| - |Zi,j - Zi,j-1|).^2
    gdl_x = abs(X(2:end,:,:,:) - X(1:end-1,:,:,:)) - abs(Z(2:end,:,:,:) - Z(1:end-1,:,:,:));
    gdl_y = abs(X(:,2:end,:,:) - X(:,1:end-1,:,:)) - abs(Z(:,2:end,:,:) - Z(:,1:end-1,:,:));
    if nargin <= 3
        diff = (X - Z) .^ 2;
        Y = 0.5 * (sum(diff(:)); % + sum(gdl_x(:).^2) + sum(gdl_y(:).^2));
    else
        Y = (X - Z); % L2
        % using |x|' = x ./ |x|; 
        Y(2:end,:,:,:) = Y(2:end,:,:,:) + gdl_lambda * gdl_x.* sign(X(2:end,:,:,:) - X(1:end-1,:,:,:));
        Y(1:end-1,:,:,:) = Y(1:end-1,:,:,:) -  gdl_lambda * gdl_x.* sign(X(2:end,:,:,:) - X(1:end-1,:,:,:));
        Y(:,2:end,:,:) = Y(:,2:end,:,:) + gdl_lambda * gdl_y .* sign(X(:,2:end,:,:) - X(:,1:end-1,:,:));
        Y(:,1:end-1,:,:) = Y(:,1:end-1,:,:) - gdl_lambda * gdl_y .* sign(X(:,2:end,:,:) - X(:,1:end-1,:,:));
        Y = Y * dzdy;
    end
end
