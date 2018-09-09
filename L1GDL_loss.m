function Y = L1GDL_loss(X, Z,gdl_lambda, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L1 (Charbonnier) loss function
%       forward : Y = vllab_nn_L1_loss(X, Z)
%       backward: Y = vllab_nn_L1_loss(X, Z, dzdy)
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

    eps = 1e-6;
%     gdl_lambda = 0.1;
    d = X - Z;
    e = sqrt( d.^2 + eps );
    
    gdl_x = abs(X(2:end,:,:,:) - X(1:end-1,:,:,:)) - abs(Z(2:end,:,:,:) - Z(1:end-1,:,:,:));
    gdl_y = abs(X(:,2:end,:,:) - X(:,1:end-1,:,:)) - abs(Z(:,2:end,:,:) - Z(:,1:end-1,:,:));
    
    e_gdl_x = sqrt(gdl_x .^2 + eps);
    e_gdl_y = sqrt(gdl_y .^2 + eps);
    
    if nargin <= 3
%         Y = sum(e(:)) + sum(e_gdl_x(:)) + sum(e_gdl_y(:));
        Y = sum(e(:)); % for statistic (to compare with L1 only)
    else
        Y = d ./ e;
%         Y(2:end,:,:,:) = Y(2:end,:,:,:) + gdl_lambda * gdl_x ./ e_gdl_x .* (X(2:end,:,:,:) - X(1:end-1,:,:,:)) ./ max(abs(X(2:end,:,:,:) - X(1:end-1,:,:,:)),1e-6);
%         Y(1:end-1,:,:,:) = Y(1:end-1,:,:,:) - gdl_lambda * gdl_x ./ e_gdl_x .* (X(2:end,:,:,:) - X(1:end-1,:,:,:)) ./ max(abs(X(2:end,:,:,:) - X(1:end-1,:,:,:)),1e-6);
%         Y(:,2:end,:,:) = Y(:,2:end,:,:) + gdl_lambda * gdl_y ./ e_gdl_y .* (X(:,2:end,:,:) - X(:,1:end-1,:,:)) ./ max(abs(X(:,2:end,:,:) - X(:,1:end-1,:,:)),1e-6);
%         Y(:,1:end-1,:,:) = Y(:,1:end-1,:,:) - gdl_lambda * gdl_y ./ e_gdl_y .* (X(:,2:end,:,:) - X(:,1:end-1,:,:)) ./ max(abs(X(:,2:end,:,:) - X(:,1:end-1,:,:)),1e-6);
        
        Y(2:end,:,:,:) = Y(2:end,:,:,:) + gdl_lambda * gdl_x ./ e_gdl_x .* sign(X(2:end,:,:,:) - X(1:end-1,:,:,:));
        Y(1:end-1,:,:,:) = Y(1:end-1,:,:,:) - gdl_lambda * gdl_x ./ e_gdl_x .* sign(X(2:end,:,:,:) - X(1:end-1,:,:,:));
        Y(:,2:end,:,:) = Y(:,2:end,:,:) + gdl_lambda * gdl_y ./ e_gdl_y .* sign(X(:,2:end,:,:) - X(:,1:end-1,:,:));
        Y(:,1:end-1,:,:) = Y(:,1:end-1,:,:) - gdl_lambda * gdl_y ./ e_gdl_y .* sign(X(:,2:end,:,:) - X(:,1:end-1,:,:));
        Y = Y .* dzdy;
    end
    
end
