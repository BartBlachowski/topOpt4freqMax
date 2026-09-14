function cfg=config(runMode,olhoffVariant)
%CONFIG One explicit configuration for smoke and production operation.
arguments
    runMode (1,:) char {mustBeMember(runMode,{'smoke','production'})}
    olhoffVariant (1,:) char {mustBeMember(olhoffVariant,{'lp','mma','both'})}
end
p=iefinal.paths();m=jsondecode(fileread(p.manifest));
cfg=struct('run_mode',runMode,'olhoff_variant',olhoffVariant,'volume_fraction',.5, ...
    'B_ref',3200,'P_primary',100,'P_sensitivity',[50 200], ...
    'quality_levels',[.98 .99 .995],'threads',1,'timing_repetitions',3, ...
    'timing_warmups',1,'candidate','C','manifest',m);
if strcmp(runMode,'production')
    cfg.meshes=double(m.production_meshes);
    cfg.reference_horizon=3200;
else
    % Short numerical plumbing only. Reference-length qualification is a
    % separate 96x12/H3200 integration test and is never confused with this.
    % Verification never drops below the 160x20 production floor: the smoke is
    % made cheap by cutting the iteration cap to 3, never by coarsening the mesh.
    cfg.meshes=[160 20];cfg.reference_horizon=3;
end
cfg.methods=iefinal.method_plan(olhoffVariant);
end
