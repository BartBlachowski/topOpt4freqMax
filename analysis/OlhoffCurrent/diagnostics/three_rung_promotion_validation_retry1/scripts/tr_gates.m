function out = tr_gates()
%TR_GATES  Repository finalization gate G1-G5 over every load-bearing study.
%
%   Reports per-study G1..G5 and enumerates the missing / mismatched entries so
%   the three provenance classes (A regenerated container, B remote-not-local,
%   C stale hash bookkeeping) can be told apart rather than collapsed.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root);
guard = olhoffcurrent_paths(); %#ok<NASGU>

D = fullfile(root,'diagnostics');
S = {'two_branch_controller_validation','three_rung_architecture', ...
     'three_rung_resolution_240','two_branch_maturity_240'};

out = struct('study',{},'ok',{},'G1',{},'G2',{},'G3',{},'G4',{},'G5',{}, ...
             'missing',{},'mismatched',{});
for i = 1:numel(S)
    st = olhoffcurrent_finalization_gate(fullfile(D,S{i}),'Verbose',false);
    out(end+1) = struct('study',S{i},'ok',st.ok, ...
        'G1',st.gates.G1,'G2',st.gates.G2,'G3',st.gates.G3, ...
        'G4',st.gates.G4,'G5',st.gates.G5, ...
        'missing',{st.missing},'mismatched',{st.mismatched}); %#ok<AGROW>
    fprintf('%-36s ok=%d G1=%d G2=%d G3=%d G4=%d G5=%d | missing=%d mismatched=%d\n', ...
        S{i}, st.ok, st.gates.G1,st.gates.G2,st.gates.G3,st.gates.G4,st.gates.G5, ...
        numel(st.missing), numel(st.mismatched));
    for j=1:numel(st.missing),    fprintf('    MISSING    %s\n', st.missing{j}); end
    for j=1:numel(st.mismatched), fprintf('    MISMATCH   %s\n', st.mismatched{j}); end
end

fid = fopen(fullfile(study,'evidence','gates.json'),'w');
c = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));
end
