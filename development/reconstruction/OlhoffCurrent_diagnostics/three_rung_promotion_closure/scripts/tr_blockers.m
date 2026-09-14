function out = tr_blockers()
%TR_BLOCKERS  Enumerate EVERY outstanding promotion-level provenance defect,
%   and classify each by what would actually close it.
%
%   The point is to answer a question the phase list does not ask: once the
%   C240 trajectory arrives, does promotion unblock?  It does not, and the
%   remaining items are named here rather than discovered later.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

S = {'two_branch_controller_validation','three_rung_architecture', ...
     'three_rung_resolution_240','three_rung_promotion_validation_retry1'};

out = struct('study',{},'ok',{},'G1',{},'G2',{},'G3',{},'G4',{},'G5',{}, ...
             'evMissing',{},'evMismatch',{},'hashMismatch',{});
for i = 1:numel(S)
    sd = fullfile(root,'diagnostics',S{i});
    g  = olhoffcurrent_finalization_gate(sd,'Verbose',false);
    ev = olhoffcurrent_evidence_gate(sd,'Verbose',false,'RepoRoot',repo);
    miss = {}; mism = {};
    for k = 1:numel(ev.artifacts)
        it = ev.artifacts(k);
        if strcmp(it.status,'REQUIRED_MISSING'),      miss{end+1} = it.path; end %#ok<AGROW>
        if strcmp(it.status,'REQUIRED_HASH_MISMATCH'),mism{end+1} = it.path; end %#ok<AGROW>
    end
    out(end+1) = struct('study',S{i},'ok',g.ok,'G1',g.gates.G1,'G2',g.gates.G2, ...
        'G3',g.gates.G3,'G4',g.gates.G4,'G5',g.gates.G5, ...
        'evMissing',{miss},'evMismatch',{mism},'hashMismatch',{g.mismatched}); %#ok<AGROW>
end

% ---- the distinct underlying artifacts, deduplicated -------------------
allMiss = {}; allMism = {}; allHash = {};
for i = 1:numel(out)
    allMiss = [allMiss, out(i).evMissing];   %#ok<AGROW>
    allMism = [allMism, out(i).evMismatch];  %#ok<AGROW>
    allHash = [allHash, out(i).hashMismatch];%#ok<AGROW>
end
out(1).distinctMissing  = unique(allMiss);
out(1).distinctMismatch = unique(allMism);
out(1).distinctHash     = unique(allHash);

fid = fopen(fullfile(study,'evidence','blockers.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== load-bearing study gates ==\n');
for i = 1:numel(out)
    fprintf('%-42s ok=%d G1=%d G2=%d G3=%d G4=%d G5=%d\n', out(i).study, out(i).ok, ...
        out(i).G1,out(i).G2,out(i).G3,out(i).G4,out(i).G5);
end
fprintf('\n== distinct REQUIRED_MISSING (transfer needed) ==\n');
for k=1:numel(out(1).distinctMissing), fprintf('  %s\n', out(1).distinctMissing{k}); end
fprintf('\n== distinct REQUIRED_HASH_MISMATCH (container digest / stale) ==\n');
for k=1:numel(out(1).distinctMismatch), fprintf('  %s\n', out(1).distinctMismatch{k}); end
fprintf('\n== distinct FINAL_SHA256 self-verify mismatches ==\n');
for k=1:numel(out(1).distinctHash), fprintf('  %s\n', out(1).distinctHash{k}); end
end
