function cp_dumpcfg()
%CP_DUMPCFG  Write EFFECTIVE_CONFIG.json from the RUNTIME-resolved configuration.
%
%   Resolves the canary configuration for both authorized meshes through the
%   real chain (cp_config -> tr_config -> cv_config -> olh.config.resolve) and
%   writes every schema leaf, the resolved hash and the model's actual free-DOF
%   count.  This replaces the earlier offline PREDICTION with the measurement.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','three_rung_promotion_validation_retry1','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

out = struct();
out.status = 'RUNTIME_RESOLVED';
out.resolved_by = 'olh.config.resolve via cp_config -> tr_config -> cv_config';
out.matlab = version;
out.when = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss'));
out.preset = 'duOlhoffFrozenM4 (olhoffcurrent_preset().upstreamPreset)';
out.production_preset_promoted_to_three_rung = false;

meshes = [480 60; 800 100];
for k = 1:size(meshes,1)
    nx = meshes(k,1); ny = meshes(k,2);
    cfg = cp_config(nx, ny);
    mdl = model2D(olh.config.toLegacy(cfg));
    key = sprintf('x%dx%d', nx, ny);
    e = struct();
    e.config_hash = olhoffcurrent_config_hash(cfg);
    e.NE = mdl.nele;
    e.free_DOF = mdl.ndof - numel(mdl.fixed);
    e.free_DOF_closed_form = 2*(nx+1)*(ny+1) - 4;
    e.rminEl = olh.config.getPath(cfg,'filter.radiusPhysical') / ...
               (olh.config.getPath(cfg,'domain.b')/ny);
    e.effective_config = rmfield(cfg, 'provenance');
    e.provenance = cfg.provenance;
    out.meshes.(key) = e;
    fprintf('[cp_dumpcfg] %dx%d  hash=%s  NE=%d  freeDOF=%d (closed form %d)  rminEl=%.6g\n', ...
        nx, ny, e.config_hash, e.NE, e.free_DOF, e.free_DOF_closed_form, e.rminEl);
end

f = fullfile(study,'EFFECTIVE_CONFIG.json');
fid = fopen(f,'w'); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[cp_dumpcfg] wrote %s\n', f);
end
