function [e, thresh] = edge(a, varargin)
% MATLAB-compatible default Sobel edge detector for Octave.
% ROAST calls edge(BW) with no arguments, i.e. MATLAB's 'sobel' method with
% an automatic threshold and directional thinning.  Octave's image-package
% edge() rejects logical input and its simple_thinning() lacks MATLAB's
% gradient-direction gate, so this reproduces MATLAB's algorithm:
%   op     = fspecial('sobel')/8
%   bx,by  = correlation of the image with op' and op ('replicate' padding)
%   b      = bx.^2 + by.^2,  cutoff = 4*mean(b(:))
%   thin   : keep a pixel only if b is a local maximum along the dominant
%            gradient direction (>= towards one neighbour, > towards the other)
if ~isempty(varargin)
    error('edge shim: only the default sobel call used by ROAST is supported');
end
a = double(a);
op = [1 2 1; 0 0 0; -1 -2 -1] / 8;
bx = imfilter(a, op', 'replicate');
by = imfilter(a, op,  'replicate');
b  = bx .* bx + by .* by;
cutoff = 4 * mean(b(:));
thresh = sqrt(cutoff);
[r, c] = size(b);
ninfC = -Inf(r, 1);
ninfR = -Inf(1, c);
left  = [ninfC, b(:, 1:end-1)];
right = [b(:, 2:end), ninfC];
up    = [ninfR; b(1:end-1, :)];
down  = [b(2:end, :); ninfR];
bxs = bx .* bx;
bys = by .* by;
alongCols = (bxs >= bys) & (b >= left) & (b > right);
alongRows = (bys >= bxs) & (b >= up)   & (b > down);
e = (b > cutoff) & (alongCols | alongRows);
end
