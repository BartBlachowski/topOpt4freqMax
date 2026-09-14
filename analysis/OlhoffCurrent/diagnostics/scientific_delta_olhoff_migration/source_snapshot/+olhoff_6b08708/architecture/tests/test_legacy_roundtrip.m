function nFail = test_legacy_roundtrip()
%TEST_LEGACY_ROUNDTRIP  toLegacy(fromLegacy(x)) must reproduce x.
%
%   Run over EVERY legacy configuration this project has ever saved: the frozen
%   TMA configs, every *.config.mat beside every audit result, the twelve anchor
%   configs, and algo/defaultCfg.  If the adapter cannot round-trip a real
%   historical config, the compatibility layer is not safe to rely on.

root = '/Users/piotrek/Programming/Matlab/Olhoff';
cases = {};

% frozen source of truth
L = load(fullfile(root,'audit_m4_topology_restoration','baseline','tree', ...
        'audit_termination_mesh_admission','runs','TMA_FROZEN_CFGS.mat'));
for k = 1:numel(L.CFG)
    cases(end+1,:) = {sprintf('TMA_FROZEN_CFGS(%d)',k), L.CFG(k).cfg}; %#ok<AGROW>
end

% every saved run config in the tree
d = dir(fullfile(root,'audit_*','**','*.mat.config.mat'));
for k = 1:numel(d)
    C = load(fullfile(d(k).folder, d(k).name));
    cases(end+1,:) = {d(k).name, C.cfg}; %#ok<AGROW>
end

% the anchors
addpath(fullfile(root,'architecture','anchors','code'));
labels = {'A1_frozen160','A2_mature160','A3_r1ladder160','A4_nodescent160', ...
          'A5_pcont160','A6_pdecoupled160','A7_massp160','A8_projidentity160', ...
          'A9_projection160','A10_binarydiag160','A11_maxnorm160','A12_rhovar160'};
for k = 1:numel(labels)
    cases(end+1,:) = {labels{k}, anchorCfg(labels{k})}; %#ok<AGROW>
end

% the superseded defaults file
cases(end+1,:) = {'algo/defaultCfg.m', defaultCfg()};

fprintf('=== legacy round-trip over %d configurations ===\n', size(cases,1));
nFail = 0;
IGNORE = {'mmasubPath'};
for k = 1:size(cases,1)
    name = cases{k,1};  flat = cases{k,2};
    try
        canon = olh.config.fromLegacy(flat);
        [canon, w] = olh.config.validate(canon);   %#ok<ASGLU>
        back  = olh.config.toLegacy(canon);
    catch e
        fprintf('  FAIL  %-42s %s: %s\n', name, e.identifier, e.message);
        nFail = nFail + 1;  continue
    end
    % rminEl is a DERIVED field whenever rminPhys is set: olhoffOpt overwrites
    % it with rminPhys/(b/nely) before building the filter, so a stored rminEl
    % is dead input.  Compare the EFFECTIVE radius the filter actually receives.
    % (Verified for T800: stored 6 and derived 6 are bitwise identical.)
    flat = local_effectiveRadius(flat);
    back = local_effectiveRadius(back);

    bad = {};
    f = setdiff(fieldnames(flat), IGNORE);
    for j = 1:numel(f)
        if ~isfield(back, f{j})
            bad{end+1} = sprintf('%s DROPPED', f{j}); %#ok<AGROW>
        elseif ~isequaln(local_norm(flat.(f{j})), local_norm(back.(f{j})))
            bad{end+1} = sprintf('%s: %s -> %s', f{j}, ...
                local_show(flat.(f{j})), local_show(back.(f{j}))); %#ok<AGROW>
        end
    end
    if isempty(bad)
        fprintf('  ok    %-42s (%d legacy fields)\n', name, numel(f));
    else
        fprintf('  FAIL  %-42s %s\n', name, strjoin(bad, ' | '));
        nFail = nFail + 1;
    end
end
fprintf('=== round-trip failures: %d ===\n', nFail);
end

function c = local_effectiveRadius(c)
if isfield(c,'rminPhys') && ~isempty(c.rminPhys) && ~isnan(c.rminPhys) && c.rminPhys > 0
    c.rminEl = c.rminPhys/(c.b/c.nely);
end
end

function v = local_norm(v)
% NaN filter radius and an empty one mean the same thing to the legacy solver.
if isnumeric(v) && isscalar(v) && isnan(v), v = []; end
if islogical(v), v = double(v); end
if isnumeric(v) && isvector(v) && ~isscalar(v), v = v(:).'; end
end

function s = local_show(v)
if ischar(v), s = ['''' v ''''];
elseif isnumeric(v) && isscalar(v), s = num2str(v);
elseif islogical(v) && isscalar(v), s = mat2str(v);
elseif isnumeric(v), s = mat2str(v);
elseif isstruct(v), s = ['struct(' strjoin(fieldnames(v)',',') ')'];
else, s = class(v);
end
end
