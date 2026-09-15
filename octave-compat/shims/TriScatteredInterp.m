classdef TriScatteredInterp
% Octave shim for MATLAB's TriScatteredInterp (default 'linear' method):
% a Delaunay triangulation of the scattered points plus barycentric linear
% interpolation, NaN outside the convex hull.  Octave has no equivalent that
% copes with ROAST-sized meshes, so the evaluation is delegated to the same
% Qhull/barycentric routine (scipy) through a helper process.
    properties
        P = [];
        V = [];
    end
    methods
        function obj = TriScatteredInterp(P, V, varargin)
            if nargin >= 2
                obj.P = double(P);
                obj.V = double(V(:));
            end
        end
        function out = subsref(obj, s)
            if numel(s) == 1 && strcmp(s(1).type, '()')
                out = evaluate(obj, s(1).subs{:});
            else
                out = builtin('subsref', obj, s);
            end
        end
        function out = evaluate(obj, xi, yi, zi)
            shim = fileparts(mfilename('fullpath'));
            work = getenv('ROAST_TSI_WORKDIR');
            if isempty(work), work = tempdir(); end
            if ~exist(work, 'dir'), mkdir(work); end
            inFile  = fullfile(work, sprintf('tsi_in_%d.mat', feature_rand()));
            outFile = [inFile '.out.mat'];
            data.P = obj.P;
            data.V = obj.V;
            % A full ndgrid(1:d1,1:d2,1:d3) query is passed as its dimensions
            % only - ROAST always interpolates onto the whole voxel grid, and
            % shipping 3 x numel(grid) doubles per call is pure overhead.
            dims = size(xi);
            if isRegularNdgrid(xi, yi, zi, dims)
                data.dims = dims;
            else
                data.Q = [xi(:), yi(:), zi(:)];
                data.qshape = dims;
            end
            save('-mat7-binary', inFile, '-struct', 'data');
            cmd = sprintf('%s %s %s %s %s', pyExe(), fullfile(shim, 'tsi_helper.py'), ...
                          inFile, outFile, fullfile(work, 'tsi_cache'));
            [st, msg] = system(cmd);
            if st ~= 0
                delete(inFile);
                error('TriScatteredInterp shim failed:\n%s', msg);
            end
            r = load(outFile);
            out = r.R;
            delete(inFile); delete(outFile);
        end
    end
end

function p = pyExe()
p = getenv('ROAST_PYTHON');
if isempty(p), p = 'python3'; end
end

function n = feature_rand()
n = round(rand() * 1e9);
end

function tf = isRegularNdgrid(xi, yi, zi, dims)
tf = false;
if numel(dims) ~= 3, return; end
if ~isequal(size(yi), dims) || ~isequal(size(zi), dims), return; end
tf = isequal(xi(:,1,1)', 1:dims(1)) && isequal(squeeze(yi(1,:,1))', (1:dims(2))') ...
     && isequal(squeeze(zi(1,1,:))', (1:dims(3))') ...
     && xi(end,1,1) == dims(1) && yi(1,end,1) == dims(2) && zi(1,1,end) == dims(3);
end
