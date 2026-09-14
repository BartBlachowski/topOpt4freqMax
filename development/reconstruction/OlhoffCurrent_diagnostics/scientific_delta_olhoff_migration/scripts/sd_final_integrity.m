function sd_final_integrity()
[P, guard] = sd_use_target(); %#ok<ASGLU>
man = olhoffcurrent_source_manifest();
[~, h] = system('git -C /Users/piotrek/Programming/Matlab/Olhoff rev-parse HEAD');
[~, d] = system('git -C /Users/piotrek/Programming/Matlab/Olhoff diff | shasum -a 256');
[~, t] = system('git -C /Users/piotrek/Programming/topOpt4freqMax rev-parse HEAD');
[~, s] = system('git -C /Users/piotrek/Programming/topOpt4freqMax status --porcelain');
out = struct('when', datestr(now,'yyyy-mm-ddTHH:MM:SS'), 'impl_manifest_ok', man.ok, 'impl_tree', man.treeHash, ...
    'source_head', strtrim(h), 'source_worktree_diff_sha256', strtrim(d), 'target_head', strtrim(t), 'target_status', strtrim(s));
fid = fopen(fullfile(P.eval,'final_integrity.json'),'w'); fprintf(fid,'%s\n',jsonencode(out,'PrettyPrint',true)); fclose(fid);
disp(out)
end
