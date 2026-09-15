function varargout = computer(varargin)
% Octave shim for the ONE MATLAB-ism ROAST needs: computer('arch').
% Octave returns 'gnu-linux-x86_64' there, while ROAST switches on MATLAB's
% 'glnxa64' / 'win64' / 'maci64'.
% The no-argument form deliberately keeps Octave's native canonical host
% string ('x86_64-pc-linux-gnu'): iso2mesh's getexeext() tests it with
% regexp(computer,'86_64') when running under Octave, and MATLAB's
% 'GLNXA64' would fail that test.
if nargin >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'arch')
    if ispc(),      varargout{1} = 'win64';
    elseif ismac(), varargout{1} = 'maci64';
    else,           varargout{1} = 'glnxa64';
    end
    return
end
varargout{1} = __octave_config_info__('canonical_host_type');
if nargout > 1, varargout{2} = 2^48-1; end
if nargout > 2, varargout{3} = 'L'; end
end
