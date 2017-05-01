function plot_feature_conv(id, img, pausesec, tt, varargin)
   figure(id);
   [w,h,c] = size(img);
   a = min(c,90);
   feat = reshape(img, w, h, 1, c);
   imdisp(feat(:,:,:,1:a), 'Border', [0.1, 0.1]);
   hold on;
   title(tt);
   colors = ['r', 'b', 'k', 'y'];
   if nargin > 4
      for l = 1:nargin - 4
          bbox = varargin{l};
          for i=1:size(bbox,1)
              rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]), 'EdgeColor', colors(mod(l, length(colors))));
              hold on;
          end
      end
   end
   hold off;
   pause(pausesec);
end
