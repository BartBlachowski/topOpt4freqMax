function [T,pairFile]=inventory_frozen()
%INVENTORY_FROZEN Verify classes/dimensions and retain genuine final-state pairs.
p=ie2b.paths();if ~isfolder(p.outputs),mkdir(p.outputs);end
meshes=[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100];
rows=cell(9,10);final_double=cell(8,1);final_single=cell(8,1);pair_mesh=cell(8,1);q=0;
for i=1:9
    nx=meshes(i,1);ny=meshes(i,2);mesh=sprintf('%dx%d',nx,ny);
    file=fullfile(p.repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' mesh '.mat']);info=dir(file);
    rows{i,1}=mesh;rows{i,2}=file;rows{i,3}=info.bytes;
    if info.bytes==0
        rows(i,4:10)={0,'N/A','N/A','N/A',false,false,'RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT'};
        continue
    end
    S=load(file,'res');r=S.res;ns=size(r.rho_snapshots,2);
    paired=isfield(r,'rho')&&isa(r.rho,'double')&&numel(r.rho)==nx*ny;
    rows(i,4:10)={ns,class(r.rho_snapshots),sprintf('%dx%d',size(r.rho_snapshots,1),ns),class(r.rho),true,paired,'AVAILABLE'};
    if paired
        q=q+1;final_double{q}=double(r.rho(:));final_single{q}=r.rho_snapshots(:,end);pair_mesh{q}=mesh;
        assert(isequal(single(r.rho(:)),r.rho_snapshots(:,end)),'ie2b:FrozenFinalPair','Final pair cast mismatch at %s.',mesh);
    end
end
T=cell2table(rows,'VariableNames',{'mesh','source_file','file_bytes','stored_states','snapshot_class','snapshot_dimensions', ...
    'returned_final_class','snapshot_original_double_recoverable','paired_final_double_single','status'});
writetable(T,fullfile(p.phase2b,'evidence_inventory.csv'));
pairFile=fullfile(p.runs,'frozen_production_final_pairs.mat');save(pairFile,'final_double','final_single','pair_mesh','-v7.3');
end
