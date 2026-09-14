function cv_export(P, file)
%CV_EXPORT  Write the per-iteration telemetry table (Phase 12) as CSV.
cols = {'outer','omega1','omega2','gap12','volume','volErr','Mnd','gray','mid', ...
        'move','stage','moveChanged','descent','beta','betaStallRel','betaStallFires', ...
        'prodStageShadow','prodMoveShadow','prodTol','prodStopRaw','prodSettled','prodStopAdmit', ...
        'exA','exB','exE','exNA','exNB','exDecl','exAmp','exCos','exNet','exMedcos','exMednet', ...
        'exTol','exStageStart','cosT','net_ratio','cosT_unsat','net_ratio_unsat', ...
        'boundFrac','revFrac','maxAbs','ratio','l2','rms','stepNorm','path_W','net_W', ...
        'nInner','cumInner','innerConv','multN','multJ','degen','tOuter'};
n = numel(P.outer);
M = nan(n, numel(cols));
for j = 1:numel(cols)
    if isfield(P, cols{j})
        v = P.(cols{j});
        if numel(v) == n, M(:,j) = double(v(:)); end
    end
end
fid = fopen(file,'w');
fprintf(fid, '%s\n', strjoin(cols, ','));
fmt = [repmat('%.17g,',1,numel(cols)-1) '%.17g\n'];
fprintf(fid, fmt, M.');
fclose(fid);
end
