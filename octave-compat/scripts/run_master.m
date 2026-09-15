warning('off','all');
more off;
pkg load image;
pkg load statistics;
addpath(fullfile(pwd, 'octave-compat', 'shims'));
try
  tk = available_graphics_toolkits();
  if any(strcmp(tk,'qt')), graphics_toolkit('qt');
  elseif any(strcmp(tk,'fltk')), graphics_toolkit('fltk');
  else, graphics_toolkit('gnuplot'); end
catch
end
set(0,'DefaultFigureVisible','off');
printf('=== toolkit: %s ===\n', graphics_toolkit());
t0 = tic;
try
  roast;
  printf('\n=== ROAST RETURNED OK ===\n');
catch ME
  printf('\n=== ROAST ERROR: %s ===\n', ME.message);
  if isfield(ME,'stack')
    for k=1:min(numel(ME.stack),12)
      printf('   at %s line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
  end
end
printf('=== ELAPSED %.1f s ===\n', toc(t0));
