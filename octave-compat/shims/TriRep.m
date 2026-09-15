function tr = TriRep(tri, x, y, z)
% Octave shim for MATLAB's TriRep. ROAST only ever builds one and hands it
% straight to freeBoundary(), so a plain record of the connectivity and the
% points is enough.
if nargin >= 4
    x = [x(:), y(:), z(:)];
end
tr = struct('Triangulation', tri, 'X', x);
end
