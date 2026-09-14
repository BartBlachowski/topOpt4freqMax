function [C, expOut, R2, nValid] = fit_complexity_model(Ne, tTotal_all, mode, fixedExp)
% FIT_COMPLEXITY_MODEL  Fit T(N_e) = C * N_e^exp per method.
%
%   [C, expOut, R2, nValid] = fit_complexity_model(Ne, tTotal_all, 'free')
%       Both C and exp are estimated by least-squares linear regression on
%       log(T) = log(C) + exp*log(N_e) (2 dof). R2 is computed on log(T).
%
%   [C, expOut, R2, nValid] = fit_complexity_model(Ne, tTotal_all, 'fixed', fixedExp)
%       exp is held fixed at fixedExp; only C is estimated, by
%       linear-space least squares minimizing sum((T - C*N_e^exp)^2)
%       (i.e. absolute run-time error, not log/relative error). R2 is
%       computed on T directly, so it is directly comparable to the
%       linear-axis plot. Closed form: C = (b'*T) / (b'*b) with
%       b = N_e.^fixedExp.
%
% Ne         : [nRes x 1] number of elements (nelx*nely)
% tTotal_all : [nRes x nMethods] total run time (s)
%
% Returns [1 x nMethods] row vectors. Entries are NaN (C, expOut, R2)
% where too few valid points are available (< 2 for 'free', < 1 for
% 'fixed'); nValid always reports the number of valid points used.

nMethods = size(tTotal_all, 2);
C      = NaN(1, nMethods);
expOut = NaN(1, nMethods);
R2     = NaN(1, nMethods);
nValid = zeros(1, nMethods);

for m = 1:nMethods
    validMask = isfinite(tTotal_all(:,m)) & tTotal_all(:,m) > 0 & Ne > 0;
    n = sum(validMask);
    nValid(m) = n;

    switch mode
        case 'free'
            if n < 2
                continue;
            end
            logNe = log(Ne(validMask));
            logT  = log(tTotal_all(validMask, m));

            A = [logNe, ones(n, 1)];
            coeffs = A \ logT;      % least-squares solution [exp; log(C)]
            expOut(m) = coeffs(1);
            C(m)      = exp(coeffs(2));

            logTHat = A * coeffs;
            ssRes = sum((logT - logTHat).^2);
            ssTot = sum((logT - mean(logT)).^2);

        case 'fixed'
            if n < 1
                continue;
            end
            NeValid = Ne(validMask);
            Tvalid  = tTotal_all(validMask, m);

            expOut(m) = fixedExp;
            basis  = NeValid .^ fixedExp;
            C(m)   = (basis' * Tvalid) / (basis' * basis);

            THat  = C(m) * basis;
            ssRes = sum((Tvalid - THat).^2);
            ssTot = sum((Tvalid - mean(Tvalid)).^2);

        otherwise
            error('fit_complexity_model:UnknownMode', 'Unknown mode "%s".', mode);
    end

    if ssTot > 0
        R2(m) = 1 - ssRes / ssTot;
    else
        R2(m) = 1;
    end
end
end
