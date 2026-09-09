function setup_roast()
% setup_roast()
%
% Put the ROAST MATLAB implementation and its bundled third-party libraries
% on the MATLAB path.
%
% Run MATLAB from the root directory of the ROAST repository (the directory
% that contains example/, data/ and lib/), then:
%
%   addpath('matlab'); setup_roast;
%   roast
%
% All data files (data/capInfo.xlsx, data/elec72.loc, data/eTPM.nii) and the
% example MRIs are resolved relative to the current directory, so the working
% directory must stay at the repository root.

thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);

addpath(genpath(thisDir));
addpath(genpath(fullfile(rootDir,'lib')));

if ~exist(fullfile(pwd,'data','capInfo.xlsx'),'file')
    warning(['ROAST expects to be run from the repository root (' rootDir ...
        '), as data and example files are resolved relative to the current directory.']);
end
