function outFile = run_native_development(nelx,nely)
%RUN_NATIVE_DEVELOPMENT Non-stopping telemetry run for detector development.
% Audit infrastructure only: the frozen reproduction tree is not modified.

if nargin < 2
    nelx=240; nely=30;
end
here=fileparts(mfilename('fullpath'));
repoRoot=fileparts(fileparts(here));
addpath(fullfile(repoRoot,'Matlab','reproduction2007','runner'));
addpath(here);
guard=repro2007_paths(); %#ok<NASGU>

[cfg,meta]=repro2007_config('fig3a_best');
cfg.nelx=nelx;
cfg.nely=nely;
cfg.verbose=false;
opts=struct('run_label',sprintf('development_%dx%d',nelx,nely), ...
    'store_density_every',1,'detector_enabled',false, ...
    'detector_active_stop',false);

fprintf('Starting non-stopping development run %dx%d, maxOuter=%d\n', ...
    nelx,nely,cfg.maxOuter);
res=olhoffOptTelemetry(cfg,opts);
identity=struct('applicable',nelx==240&&nely==30,'passed',NaN, ...
    'fields',struct(),'max_abs',struct(),'baseline',meta.baseline_artifact);
if identity.applicable
    baseline=load(fullfile(repoRoot,'Matlab','reproduction2007', ...
        'baseline','lp240_rmin1.3.mat'),'res');
    base=baseline.res;
    fields={'rho','omega','lambda','nOuter','modeTable'};
    for i=1:numel(fields)
        f=fields{i};
        identity.fields.(f)=isequaln(res.(f),base.(f));
        if isnumeric(res.(f))
            identity.max_abs.(f)=max(abs(double(res.(f)(:))-double(base.(f)(:))));
        end
    end
    histFields={'omega','N','beta','nInner','dxOuter','vol','degen', ...
        'multJ','innerConv','cumInner'};
    for i=1:numel(histFields)
        f=histFields{i}; key=['hist_' f];
        identity.fields.(key)=isequaln(res.hist.(f),base.hist.(f));
        identity.max_abs.(key)=max(abs(double(res.hist.(f)(:))-double(base.hist.(f)(:))));
    end
    values=struct2cell(identity.fields);
    identity.passed=all(cellfun(@(x)isequal(x,true),values));
    if ~identity.passed
        error('run_native_development:IdentityFailure', ...
            'Telemetry mirror changed at least one frozen numerical result.');
    end
end

resultDir=fullfile(here,'results');
if exist(resultDir,'dir')~=7, mkdir(resultDir); end
outFile=fullfile(resultDir,sprintf('development_%dx%d.mat',nelx,nely));
save(outFile,'res','identity','-v7.3');
fprintf('Saved %s (nOuter=%d, wallclock=%.1fs)\n',outFile,res.nOuter,res.wallclock);
if identity.applicable
    fprintf('Frozen trajectory identity: %s\n',string(identity.passed));
end
end
