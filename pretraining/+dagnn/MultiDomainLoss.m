classdef MultiDomainLoss < dagnn.Loss
%LossSmoothL1  Smooth L1 loss
%  `LossSmoothL1.forward({x, x0, w})` computes the smooth L1 distance 
%  between `x` and `x0`, weighting the elements by `w`.
%
%  Here the smooth L1 loss between two vectors is defined as:
%
%     Loss = sum_i f(x_i - x0_i) w_i.
%
%  where f is the function (following the Faster R-CNN definition):
%
%              { 0.5 * sigma^2 * delta^2,         if |delta| < 1 / sigma^2,
%   f(delta) = {
%              { |delta| - 0.5 / sigma^2,         otherwise.
%
%  In practice, `x` and `x0` can pack multiple instances as 1 x 1 x C
%  x N arrays (or simply C x N arrays).

  methods
   function outputs = forward(obj, inputs, params)
      data = inputs{1};
      K = inputs{3};
      data = data(:,:, 2*K-1:2*K, :);
      outputs{1} = vl_nnloss(data, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
      % Accumulate loss statistics.
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
 
   end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      data = inputs{1};
      K = inputs{3};
      data = data(:, :, 2*K-1:2*K, :);
      x = vl_nnloss(data, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
      if isa(x,'gpuArray')
          der = gpuArray.zeros(size(inputs{1}), classUnderlying(inputs{1}));
      else
          der = zeros(size(inputs{1}), classUnderlying(inputs{1}));
      end
      der(:,:, 2*K-1:2*K, :)  = x; 
      derInputs{1} = der;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
 
   end

    function obj = MultiDomainLoss(varargin)
      obj.load(varargin) ;
      obj.loss = 'softmaxlog';
    end
  end
end
