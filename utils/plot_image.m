function plot_image(id, img, pausesec, varargin)
   figure(id);
   imshow(img);
   hold on;
   colors = ['r', 'b', 'k', 'y'];
   if nargin > 3
      for l = 1:nargin - 3
          bbox = varargin{l};
          for i=1:size(bbox,1)
              rectangle('Position',round([bbox(i,1),bbox(i,2), bbox(i,3) - bbox(i,1) + 1, bbox(i,4) - bbox(i,2) + 1]), 'EdgeColor', colors(mod(l, length(colors))));
              hold on;
          end
      end
   end
   pause(pausesec);
   hold off;
end
