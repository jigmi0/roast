function viewMRI(T1,T2,mri2mni)
% viewMRI(T1,T2,mri2mni)
% 
% Show the T1 (and the T2, when one was given) in the slice viewer. mri2mni
% maps voxel coordinates to MNI space, so the viewer can report both.
% 
% (c) Yu (Andy) Huang
% yhuang16@citymail.cuny.edu
% July 2025

t1Data = load_untouch_nii(T1);
sliceshow(t1Data.img,[],'gray',[],[],'MRI: Click anywhere to navigate.',[],mri2mni); drawnow

if ~isempty(T2)
    t2Data = load_untouch_nii(T2);
    sliceshow(t2Data.img,[],'gray',[],[],'MRI: T2. Click anywhere to navigate.',[],mri2mni); drawnow
end