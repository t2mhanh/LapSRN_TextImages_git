classdef gdl_dag_loss < dagnn.Loss
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
%       
%
%   Contact:
%       
% -------------------------------------------------------------------------
  properties
    lambda = 1;
    loss_type = 'L2_GDL';
    average_by_size = 0;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      if( strcmp(obj.loss_type, 'L2_GDL') )
        outputs{1} = vl_nn_L2GDL_loss(inputs{1}, inputs{2});
        
      elseif( strcmp(obj.loss_type, 'L1_GDL') )
        outputs{1} = vl_nn_L1GDL_loss(inputs{1}, inputs{2});
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
        derInputs{1} = vl_nn_L2GDL_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      elseif( strcmp(obj.loss_type, 'L1_GDL') )
        derInputs{1} = vl_nn_L1GDL_loss(inputs{1}, inputs{2}, derOutputs{1}) ;
      else
        error('Unknown loss %s\n', obj.loss_type);
      end
      derInputs{1} = obj.lambda * derInputs{1};
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = gdl_dag_loss(varargin)
      obj.load(varargin) ;
    end
  end
end
