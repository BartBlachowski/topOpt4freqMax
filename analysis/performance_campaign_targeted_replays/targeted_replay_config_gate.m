function gate = targeted_replay_config_gate(method)
%TARGETED_REPLAY_CONFIG_GATE Build the replay from the frozen campaign profile.
% The returned replay differs only in declared diagnostic/history fields.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo,'examples','Performance'));
addpath(fullfile(repo,'analysis','three_method_parametric_study'));

switch lower(char(method))
    case 'olhoff'
        nx=640; ny=80; canonical='Olhoff';
    case 'yuksel'
        nx=800; ny=100; canonical='Yuksel';
    case {'proposed','ourapproach'}
        nx=160; ny=20; canonical='Proposed';
    otherwise
        error('targeted_replay_config_gate:UnknownMethod','Unknown method %s.',method);
end

[original, profileId] = final_campaign_config(canonical,nx,ny,here);
replay = original;
diagnostic = struct('record_history',false,'native_spectral_history',false, ...
    'extra_scalar_telemetry',false,'density_snapshot_stride',NaN, ...
    'failed_attempt_retention',false);
switch canonical
    case 'Olhoff'
        diagnostic.failed_attempt_retention=true;
    case 'Yuksel'
        replay.benchmark.record_history=true;
        diagnostic.record_history=true;
        diagnostic.extra_scalar_telemetry=true;
        diagnostic.density_snapshot_stride=10;
    case 'Proposed'
        replay.benchmark.record_history=true;
        replay.postprocessing.save_frequency_iterations=true;
        diagnostic.record_history=true;
        diagnostic.native_spectral_history=true;
end

normOriginal=strip_diagnostics(original);
normReplay=strip_diagnostics(replay);
pass=isequaln(normOriginal,normReplay);
assert(pass,'targeted_replay_config_gate:NumericalMismatch', ...
    '%s replay differs from the frozen numerical configuration.',canonical);
assert(maxNumCompThreads==1,'targeted_replay_config_gate:ThreadMismatch', ...
    'Replay requires exactly one MATLAB computation thread.');

expected=source_expectations();
for i=1:numel(expected)
    rel=expected(i).path;
    actual=file_sha256(fullfile(repo,rel));
    assert(strcmp(actual,expected(i).sha256), ...
        'targeted_replay_config_gate:SourceHashMismatch', ...
        'Frozen source changed: %s.',rel);
end

cfgDir=fullfile(here,'configurations');
if exist(cfgDir,'dir')~=7,mkdir(cfgDir);end
write_json(fullfile(cfgDir,sprintf('%s_original.json',lower(canonical))),original);
write_json(fullfile(cfgDir,sprintf('%s_replay_effective.json',lower(canonical))),replay);
write_json(fullfile(cfgDir,sprintf('%s_normalized_original.json',lower(canonical))),normOriginal);
write_json(fullfile(cfgDir,sprintf('%s_normalized_replay.json',lower(canonical))),normReplay);

gate=struct('method',canonical,'mesh',sprintf('%dx%d',nx,ny), ...
    'nelx',nx,'nely',ny,'original_profile_id',profileId, ...
    'replay_profile_id',profileId,'original',original,'replay',replay, ...
    'normalized_original',normOriginal,'normalized_replay',normReplay, ...
    'diagnostic_only_differences',diagnostic,'numerical_config_identical',pass, ...
    'pass',pass);
write_json(fullfile(cfgDir,sprintf('%s_gate.json',lower(canonical))),gate);
fprintf('CONFIG_IDENTITY_%s=PASS profile=%s\n',upper(canonical),profileId);
end

function s=strip_diagnostics(s)
if isfield(s,'benchmark')
    diagnosticFields={'enable_diagnostics','record_history'};
    for i=1:numel(diagnosticFields)
        if isfield(s.benchmark,diagnosticFields{i})
            s.benchmark=rmfield(s.benchmark,diagnosticFields{i});
        end
    end
end
if isfield(s,'postprocessing')
    diagnosticFields={'visualize_live','save_final_image','save_snapshot_image', ...
        'save_frequency_iterations','visualize_quality','visualize_modes', ...
        'visualize_topology_modes','write_correlation_table'};
    for i=1:numel(diagnosticFields)
        if isfield(s.postprocessing,diagnosticFields{i})
            s.postprocessing=rmfield(s.postprocessing,diagnosticFields{i});
        end
    end
end
end

function e=source_expectations()
paths={ ...
 'analysis/olhoff_stabilization_audit/final_campaign_profile.json', ...
 'analysis/olhoff_stabilization_audit/selected_profile.json', ...
 'analysis/three_method_parametric_study/results/profile_freeze_manifest.json', ...
 'examples/Performance/final_campaign_config.m', ...
 'analysis/olhoff_stabilization_audit/olhoffOptStabilized.m', ...
 'Matlab/reproduction2007/algo/innerLoopLP.m', ...
 'Matlab/reproduction2007/fem/eigSolve.m', ...
 'tools/Matlab/run_topopt_from_json.m', ...
 'analysis/YukselApproach/Matlab/top99neo_inertial_freq.m', ...
 'analysis/ourApproach/Matlab/topopt_freq.m'};
hashes={ ...
 '5d12bc0ae6a09d2f4df01fb38d7f483a7450b06cac67eceac30d3fab3618b610', ...
 '60fa944f4aecf34611de5413096d6a0de235eae05febbc4dd481bee5a26a67da', ...
 'b55e31d87d18e90e8c0b8d278bd4111d494610b62f253194ea16cfcf78252eca', ...
 'bd3e74673315da907e2e5d5b06ba1a8b94ce372977342bde04de8be30fe6a65c', ...
 '95240cf60f82b40f8e5e892b9eea9b20a8fd3744b5eca6fdfc8dde2698d82aec', ...
 '7724753c02f84d6009c3998f758d5b3f9c5144ad39ca6f470584a2c99e089465', ...
 'b0784ceeb15fafe164ba138b963c3ab4dc5a466fc2953c79786809edd16159cd', ...
 '7b89f42d86ef6d7974fa68565a8c0e83068fa580ca65e123380fba6c50bfbc56', ...
 '5afc3d16b4ed6af05793df461b541ed3b2ea62a6da8836f38301a9a3917e6ba2', ...
 '6d9ea66fcc27f63b7380708b5735552b5d9f2885d3e65714af572daccdae72b2'};
e=struct('path',paths,'sha256',hashes);
end

function h=file_sha256(path)
fid=fopen(path,'rb');assert(fid>=0,'Unreadable %s',path);c=onCleanup(@()fclose(fid)); %#ok<NASGU>
b=fread(fid,Inf,'*uint8');md=java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(b),md.update(b);end;d=typecast(md.digest(),'uint8');h=lower(reshape(dec2hex(d,2).',1,[]));
end

function write_json(path,value)
fid=fopen(path,'w');assert(fid>=0,'Cannot write %s',path);c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
