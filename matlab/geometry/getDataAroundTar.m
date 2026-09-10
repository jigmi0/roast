function val = getDataAroundTar(data,target,x,y,z,radius)
% val = getDataAroundTar(data,target,x,y,z,radius)
% 
% Mean of data inside a sphere of the given radius around target, ignoring
% NaNs. Used when the target voxel itself falls outside the brain mask, where
% reading that single voxel would just return NaN. x, y and z are the ndgrid
% coordinates of data.
% 
% Yu (Andy) Huang, Aug 2025

tarMsk=sqrt((x-target(1)).^2+(y-target(2)).^2+(z-target(3)).^2)<radius;
[xt,yt,zt]=ind2sub(size(tarMsk),find(tarMsk(:)==1));
val = nan(length(xt),1);
for i=1:length(xt), val(i)=data(xt(i),yt(i),zt(i)); end
val = nanmean(val);