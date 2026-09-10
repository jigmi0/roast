function [x,cvx_status] = max_l1per(f,ub,solElecNum,verbose)
% [x,cvx_status] = max_l1per(f,ub,solElecNum,verbose)
%
% Maximum intensity at the target, with an L1 bound on the total current AND on
% each electrode.
%
% As max_l1(), but no single electrode may carry more than 2*ub/solElecNum,
% which drives the solution towards a montage of exactly solElecNum electrodes.
%
% Accepts any number of targeting ROIs.
% solElecNum is the desired number of anodes+cathodes in the optimal
% solution/montage.
% ANDY 2014-10-27
% ANDY 2017-01-30

n = size(f,1);

if verbose
    cvx_begin
else
    cvx_begin quiet
end
              variable x(n);
              maximize( sum(f'*x) );
               subject to
                  norm([x;-sum(x)],1) <= 2*ub;
                  norm([x;-sum(x)],inf) <= 2*ub/solElecNum;
    cvx_end
