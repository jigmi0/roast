function save(varargin)
% Octave shim for MATLAB's save(..., '-v7.3').
% Octave 8 cannot write the HDF5-based -v7.3 container.  ROAST asks for it so
% that very large results still load in MATLAB; this model's results fit the
% v7 container, which .octaverc already makes the default format, so the flag
% is dropped and everything else is delegated to Octave's builtin save,
% evaluated in the caller so it sees the caller's variables.
args = varargin;
for i = 1:numel(args)
    if ~ischar(args{i})
        error('save shim: only char arguments are supported (got %s)', class(args{i}));
    end
end
keep = ~strcmpi(args, '-v7.3');
args = args(keep);
parts = cell(1, numel(args));
for i = 1:numel(args)
    parts{i} = ['''' strrep(args{i}, '''', '''''') ''''];
end
evalin('caller', ['builtin(''save'', ' strjoin(parts, ', ') ');']);
end
