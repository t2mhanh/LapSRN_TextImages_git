classdef GDL_loss < dagnn.Loss
% -------------------------------------------------------------------------
%   Description:
%       loss object for dagnn class
%       if using your own MatConvNet version, copy this file to [matconvnet]/matlab/+dagnn
%
%   Parameters:
%       - lambda    : weight of loss
%       - loss_type : support 'L1-GDL' or 'L2-DGL' loss
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
  properties
    lambda = 1;
    loss_type = 'L2_GDL';
    average_by_size = 0;
    gdl_lambda = 1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      if( strcmp(obj.loss_type, 'L2_GDL') )
        outputs{1} = L2GDL_loss(inputs{1}, inputs{2},obj.gdl_lambda);
        
      elseif( strcmp(obj.loss_type, 'L1_GDL') )
        outputs{1} = L1GDL_loss(inputs{1}, inputs{2}, obj.gdl_lambda);
      else
        error('Unknown loss %s\n', obj.loss_type);
      end
      outputs{1} = obj.lambda * outputs{1};
      
      if obj.average_by_size
        h = size(inputs{1}, 1);
        w = size(inputs{1}, 2);
        outputs{1} = outputs{1} / (h * w);
      end
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if( strcmp(obj.loss_type, 'L2_GDL') )
        derInputs{1} = L2GDL_loss(inputs{1}, inputs{2},obj.gdl_lambda, derOutputs{1}) ;
      elseif( strcmp(obj.loss_type, 'L1_GDL') )
        derInputs{1} = L1GDL_loss(inputs{1}, inputs{2},obj.gdl_lambda, derOutputs{1}) ;
      else
        error('Unknown loss %s\n', obj.loss_type);
      end
      derInputs{1} = obj.lambda * derInputs{1};
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = GDL_loss(varargin)
      obj.load(varargin) ;
    end
  end
end
