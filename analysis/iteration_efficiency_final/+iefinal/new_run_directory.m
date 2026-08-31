function out=new_run_directory(runMode,selector)
%NEW_RUN_DIRECTORY Create collision-proof mode/selector-isolated output.
p=iefinal.paths();stamp=char(datetime('now','Format','yyyyMMdd''T''HHmmssSSS'));
base=fullfile(p.runs,runMode,selector);out=fullfile(base,stamp);
assert(~isfolder(out),'iefinal:OutputCollision','Refusing to overwrite %s.',out);
mkdir(out);
for d={'reference','measurement','timing','analysis','tables','figures','topologies','provenance','validation'}
    mkdir(fullfile(out,d{1}));
end
end
