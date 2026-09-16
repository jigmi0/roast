1;  % force script interpretation, so the definitions below are script-local
%
% Script-local functions take precedence over both the current directory and
% the load path, which neutralises two Octave-only problems WITHOUT touching
% a single file in the ROAST checkout:
%
%   * viewMRI/viewSeg/viewElectrodes/visualizeRes are display-only and need
%     uifigure, which Octave does not have. roast.m discards their outputs,
%     so no-ops change no numbers. Equivalent to the port's show=False.
%   * lib/cvx/{functions,sedumi}/vec.m shadow Octave's core 2-argument vec(),
%     which the image package's fspecial() needs. This restores both arities.
%
function viewMRI(varargin)
end
function viewSeg(varargin)
end
function viewElectrodes(varargin)
end
function visualizeRes(varargin)
end
function y = vec(x, dim)
  if nargin < 2
    y = x(:);
  else
    y = reshape(x, [ones(1, dim-1), numel(x), 1]);
  end
end

warning('off','all'); more off;
pkg load image; pkg load statistics;
addpath(fullfile(pwd,'octave-compat','shims'));
try
  tk = available_graphics_toolkits();
  if any(strcmp(tk,'qt')), graphics_toolkit('qt'); else, graphics_toolkit('gnuplot'); end
catch
end
set(0,'DefaultFigureVisible','off');
cmd = getenv('ROAST_CMD');
printf('=== RUNNING: %s ===\n', cmd);
t0 = tic;
try
  eval(cmd);
  printf('\n=== EXAMPLE OK ===\n');
catch ME
  printf('\n=== EXAMPLE ERROR: %s ===\n', ME.message);
  for k=1:min(numel(ME.stack),6), printf('   at %s line %d\n', ME.stack(k).name, ME.stack(k).line); end
end
printf('=== ELAPSED %.1f s ===\n', toc(t0));
