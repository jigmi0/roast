function [x,cvx_status] = max_l1(f,ub,verbose)
% [x,cvx_status] = max_l1(f,ub,verbose)
%
% Maximum intensity at the target, with an L1 bound on the injected current.
%
% Maximises sum(f'*x), the field component along the desired orientation at the
% targets, subject to a total injected current of at most ub. Column n of f
% holds the lead field projected onto the desired orientation of target n.
%
% Accepts any number of targeting ROIs.
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
    cvx_end
