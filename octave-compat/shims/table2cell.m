function c = table2cell(t)
% Octave compatibility shim matching the readtable shim above.
if isstruct(t) && isfield(t, 'data')
    c = t.data;
else
    error('table2cell shim: unexpected input');
end
end
