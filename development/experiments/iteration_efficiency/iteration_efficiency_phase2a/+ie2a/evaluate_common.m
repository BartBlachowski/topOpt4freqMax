function out = evaluate_common(x, nelx, nely, volumeFraction, opts)
%EVALUATE_COMMON Candidate-C E1/E2/E3 evaluation of the actual gray field.
arguments
    x {mustBeNumeric,mustBeReal,mustBeFinite}
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    volumeFraction (1,1) double = 0.5
    opts.IncludeBinaryDiagnostic (1,1) logical = false
    opts.TechnicalMaxModes (1,1) double {mustBePositive} = Inf
    opts.InjectEigensolverFailure (1,1) logical = false
    opts.InjectInvalidEigenpairs (1,1) logical = false
    opts.InjectNonfiniteDiagnostics (1,1) logical = false
end
assert(exist('study_evaluate_design','file')==2,'ie2a:EvaluatorUnavailable', ...
    'The frozen study_evaluate_design evaluator is not on the MATLAB path.');
x=double(x(:));
out.raw=study_evaluate_design(x,nelx,nely,volumeFraction, ...
    ComputeBinaryDiagnostic=opts.IncludeBinaryDiagnostic, ...
    TechnicalMaxModes=opts.TechnicalMaxModes, ...
    InjectEigensolverFailure=opts.InjectEigensolverFailure, ...
    InjectInvalidEigenpairs=opts.InjectInvalidEigenpairs, ...
    InjectNonfiniteDiagnostics=opts.InjectNonfiniteDiagnostics);
out.Q=[out.raw.selected_omega_raw_E1,out.raw.selected_omega_raw_E2,out.raw.selected_omega_raw_E3];
out.Q_raw=out.Q; % compatibility name; values are C-selected gray structural frequencies
out.evaluator_status={out.raw.status_raw_E1,out.raw.status_raw_E2,out.raw.status_raw_E3};
out.selected_ordinal=[out.raw.selected_ordinal_raw_E1,out.raw.selected_ordinal_raw_E2,out.raw.selected_ordinal_raw_E3];
out.modal={out.raw.modal_raw_E1,out.raw.modal_raw_E2,out.raw.modal_raw_E3};
if all(strcmp(out.evaluator_status,'PASS'))&&all(isfinite(out.Q))
    out.status='PASS';
else
    out.status='STRUCTURAL_MODE_NOT_FOUND';
end
out.binary_role='ENDPOINT_MANUFACTURABILITY_TOPOLOGY_DIAGNOSTIC_EXCLUDED_FROM_Q';
if opts.IncludeBinaryDiagnostic
    out.Q_binary_endpoint_diagnostic=[localFirst(out.raw.omega_binary_E1), ...
        localFirst(out.raw.omega_binary_E2),localFirst(out.raw.omega_binary_E3)];
else
    out.Q_binary_endpoint_diagnostic=[NaN NaN NaN];
end
out.Q_binary=out.Q_binary_endpoint_diagnostic; % compatibility; never consumed by Q/reference/persistence
out.evaluator_ids={'E1','E2','E3'};
out.evaluator_source='analysis/three_method_parametric_study/study_evaluate_design.m';
out.evaluator_candidate='C';
out.modal_classifier_version='candidate_c_unanimous_v1';
out.Q_source='ACTUAL_GRAY_LOWEST_UNANIMOUS_VALID_STRUCTURAL_MODE';
end
function x=localFirst(v)
if isempty(v),x=NaN;else,x=v(1);end
end
