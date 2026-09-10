function [x,cvx_status] = ls_l1per(A,d,ub,verbose)
% [x,cvx_status] = ls_l1per(A,d,ub,verbose)
%
% Least squares with an L1 bound on the total current AND on each electrode.
%
% As ls_l1(), but no single electrode may carry more than ub/2, which spreads
% the current over more electrodes instead of concentrating it on a few.
%
% Accepts any number of targeting ROIs.
% ANDY 2014-10-27
% ANDY 2017-01-30

n = size(A,2);

if verbose
    cvx_begin
else
    cvx_begin quiet
end
              variable x(n);
              minimize( norm(A*x-d,2) );
               subject to
                  norm([x;-sum(x)],1) <= 2*ub;
                  norm([x;-sum(x)],inf) <= ub/2;
    cvx_end
