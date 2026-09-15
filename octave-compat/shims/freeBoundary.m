function [FF, XF] = freeBoundary(tr)
% Octave shim for MATLAB's TriRep/freeBoundary.
% Returns the facets referenced by exactly one simplex (the free surface),
% re-indexed into XF, the ascending-ordered subset of points they use -
% the same contract as MATLAB's freeBoundary.
t = tr.Triangulation;
X = tr.X;
if size(t, 2) == 4                       % tetrahedra -> triangular facets
    f = [t(:,[1 2 3]); t(:,[1 2 4]); t(:,[1 3 4]); t(:,[2 3 4])];
elseif size(t, 2) == 3                   % triangles -> edges
    f = [t(:,[1 2]); t(:,[1 3]); t(:,[2 3])];
else
    error('freeBoundary shim: unsupported simplex size %d', size(t,2));
end
fs = sort(f, 2);
[u, ~, ic] = unique(fs, 'rows');
counts = accumarray(ic, 1);
FFglobal = u(counts == 1, :);
nodeIdx = unique(FFglobal(:));
XF = X(nodeIdx, :);
map = zeros(max(nodeIdx), 1);
map(nodeIdx) = 1:numel(nodeIdx);
FF = map(FFglobal);
if size(FF, 2) ~= size(FFglobal, 2), FF = reshape(FF, size(FFglobal)); end
end
