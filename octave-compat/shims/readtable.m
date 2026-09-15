function t = readtable(fname, varargin)
% Octave compatibility shim for MATLAB's readtable, limited to the
% spreadsheet reads ROAST performs on capInfo.xlsx.
% The sheets carry no header row, so every row is data - the same result
% MATLAB's readtable auto-detection produces here.
% str2double (not textscan %f) is used so the doubles round-trip exactly.
sheet = '';
for i = 1:2:numel(varargin)
    if strcmpi(varargin{i}, 'Sheet'), sheet = varargin{i+1}; end
end
if isempty(sheet), error('readtable shim: ''Sheet'' must be given'); end
[~, base] = fileparts(fname);
csv = fullfile(fileparts(mfilename('fullpath')), [base '__' sheet '.csv']);
fid = fopen(csv, 'r');
if fid == -1, error('readtable shim: no cached sheet at %s', csv); end
raw = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = raw{1};
n = numel(lines);
data = cell(n, 4);
for r = 1:n
    f = strsplit(lines{r}, ',');
    data{r,1} = f{1};
    data{r,2} = str2double(f{2});
    data{r,3} = str2double(f{3});
    data{r,4} = str2double(f{4});
end
t = struct('roast_shim_table', true, 'data', {data});
end
