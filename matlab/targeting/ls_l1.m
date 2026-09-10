function [x,cvx_status] = ls_l1(A,d,ub,verbose)
% [x,cvx_status] = ls_l1(A,d,ub,verbose)
%
% Least squares with an L1 bound on the injected current.
%
% Finds the electrode currents x whose field A*x comes closest to the desired
% field d, subject to a total injected current of at most ub. The reference
% electrode carries -sum(x), so appending it to x makes the 1-norm the sum of
% all the current going in and out of the head, which is 2*ub.
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
    cvx_end
