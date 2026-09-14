function audit_volume_mma()
%AUDIT_VOLUME_MMA  Does the PENALISED volume constraint of the nested MMA let
%   the filtered volume drift?  The LP route enforces the row exactly; mmasub
%   treats it as a c=1000 penalty, so this must be measured, not assumed.
addpath('/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffRegularized/Matlab');
rc=struct('verbose',false,'formulation','olhoff','optimizer','mma', ...
    'max_outer_iterations',12,'persistence',1000);
[rho,~,info]=topopt_olhoff_regularized(160,20,.5,3,1.5,.005,'simply',rc);
v=info.history.volume;
fprintf('olhoff/mma 160x20, 12 outer iterations\n');
for k=1:numel(v),fprintf('  outer %2d  mean(rho) = %.12f  drift = %+.3e\n',k,v(k),v(k)-0.5);end
fprintf('  terminal mean(rho) = %.12f  max drift over run = %.3e\n',mean(rho),max(abs(v-0.5)));
end
