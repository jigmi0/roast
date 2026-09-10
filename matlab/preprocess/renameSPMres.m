function renameSPMres(src,tar)
% renameSPMres(src,tar)
%
% Rename the SPM12 segmentation outputs from the name of the image that was
% segmented to the name of the model, so that ROAST can tell a run that used
% the T1 alone from one that was also given a T2. Moves the six tissue maps
% c1..c6 along with the _rmask.mat and _seg8.mat files.
%
% (c) Andrew Birnbaum, Parra Lab at CCNY
%     Yu (Andy) Huang
% April 2024

[dir,srcName] = fileparts(src);
if isempty(dir), dir = pwd; end
[~,tarName] = fileparts(tar);

for t = 1:6
    movefile([dir filesep 'c' num2str(t) srcName '.nii'], ...
        [dir filesep 'c' num2str(t) tarName '.nii']);
end

movefile([dir filesep srcName '_rmask.mat'], ...
    [dir filesep tarName '_rmask.mat']);

movefile([dir filesep srcName '_seg8.mat'], ...
    [dir filesep tarName '_seg8.mat']);
