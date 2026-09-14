function verdict=classify_cell_failure(ME)
%CLASSIFY_CELL_FAILURE Separate cell-local scientific failure from integrity failure.
% A cell-local SCIENTIFIC EXECUTION failure is one where the optimization method
% itself did not produce a usable trajectory for this method/mesh: the native
% solver failed, exhausted resources, or returned a degenerate/short result.
% Those become a RUN_ERROR cell and the campaign continues.
%
% Everything else -- contract/hash mismatch, schema violation, output-isolation
% breach, trajectory-integrity or determinism failure, and any ordinary
% programming error -- is an INTEGRITY failure and stays campaign-fatal. The
% list is an explicit allowlist so a new or unforeseen error can never be
% silently reclassified as a scientific result.
arguments
    ME MException
end
id=string(ME.identifier);

% Native solver / method execution failures raised by the harness wrappers.
harnessScientific=[ ...
    "iefinal:OptimizerFailure"           % LP returned SOLVER_FAILURE
    "iefinal:MissingTrajectory"          % method produced no eligible states
    "iefinal:NonfiniteTrajectory"        % method produced a nonfinite state
    "iefinal:MissingReferenceTrajectory" % native run too short for the reference horizon
    ];

% Failures raised from inside the native numerics themselves.
solverPrefixes=[ ...
    "MATLAB:eigs:"
    "MATLAB:svds:"
    "MATLAB:decomposition:"
    "repro2007:"
    ];
solverExact=[ ...
    "MATLAB:nomem"
    "MATLAB:pmaxsize"
    "MATLAB:array:SizeLimitExceeded"
    "MATLAB:singularMatrix"
    "MATLAB:illConditionedMatrix"
    "MATLAB:posdef"
    ];

isScientific=any(id==harnessScientific)||any(id==solverExact)|| ...
    any(arrayfun(@(pfx)startsWith(id,pfx),solverPrefixes));

if isScientific
    verdict=struct('cell_local',true,'status','RUN_ERROR','class','scientific_execution');
else
    verdict=struct('cell_local',false,'status','INTEGRITY_FAILURE','class','integrity');
end
verdict.identifier=char(id);
verdict.message=ME.message;
end
