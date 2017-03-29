classdef MultiDomainLossSmoothL1 < dagnn.Loss
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

  properties
    sigma = 1.
  end

  methods
    function outputs = forward(obj, inputs, params)
      sigma2 = obj.sigma^2 ;
      K = inputs{4};
      input_p = inputs{1}(:,:,8*K - 7:8*K, :);
      delta = input_p - inputs{2} ;
      absDelta = abs(delta) ;

      linearRegion = (absDelta > 1. / sigma2) ;
      absDelta(linearRegion) = absDelta(linearRegion) - 0.5/sigma2 ;
      absDelta(~linearRegion) = 0.5 * sigma2 * absDelta(~linearRegion).^2 ;

      % Mutliply by instance weights and sum.
      outputs{1} = inputs{3}(:)' * absDelta(:) ;

      % Accumulate loss statistics.
      n = obj.numAveraged ;
      m = n + gather(sum(inputs{3}(:))) + 1e-9 ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
    % Function derivative:
    %
    %          { sigma^2 * x,             if |x| < 1 / sigma^2,
    %  f'(x) = {
    %          { sign(x),                 otherwise.
      K = inputs{4};
      input_p = inputs{1}(:,:,8*K-7:8*K,:);
      sigma2 = obj.sigma^2 ;
      delta = input_p - inputs{2} ;
      absDelta = abs(delta) ;
     
      linearRegion = (absDelta > 1. / sigma2) ;
      delta(linearRegion) = sign(delta(linearRegion));
      delta(~linearRegion) = sigma2 * delta(~linearRegion) ;
      x = inputs{3} .* delta .* derOutputs{1};
      if isa(x,'gpuArray')
          der = gpuArray.zeros(size(inputs{1}), classUnderlying(inputs{1}));
      else
          der = zeros(size(inputs{1}), 'single');
      end
      der(:,:,8*K-7:8*K,:) = x;
      derInputs = {der, [], [], []} ;
      derParams = {} ;
    end

    function obj = MultiDomainLossSmoothL1(varargin)
      obj.load(varargin) ;
      obj.loss = 'smoothl1';
    end
  end
end
