function artifacts=render_topologies(cells,rows,outputDir)
%RENDER_TOPOLOGIES Shared raw/binary accepted-state grids, outside timing.
if ~isfolder(outputDir),mkdir(outputDir);end
meshes=unique(string({rows.mesh}),'stable');artifacts=struct([]);
for im=1:numel(meshes)
    rr=rows(string({rows.mesh})==meshes(im)&[rows.P]==100&abs([rows.q]-.99)<1e-12);
    states=struct([]);s=0;
    for i=1:numel(rr)
        ci=find(arrayfun(@(c)strcmp(c.trajectory.method,rr(i).method)&& ...
            strcmp(c.trajectory.method_variant,rr(i).method_variant)&& ...
            strcmp(sprintf('%dx%d',c.trajectory.nelx,c.trajectory.nely),rr(i).mesh),cells),1);
        s=s+1;rec=localState(rr(i),cells,ci,false);if s==1,states=rec;else,states(s)=rec;end %#ok<AGROW>
        s=s+1;states(s)=localState(rr(i),cells,ci,true); %#ok<AGROW>
    end
    base=fullfile(outputDir,['accepted_q0p99_' char(meshes(im))]);
    rec=render_iteration_efficiency_topology_grid(states,base,struct('Visible','off','OverlayPolicy','supports'));
    rec=localSerializable(rec);
    if im==1,artifacts=rec;else,artifacts(im)=rec;end %#ok<AGROW>
end
end
function rec=localSerializable(rec)
%LOCALSERIALIZABLE Drop the renderer's live graphics handles from the record.
% The shared renderer returns figure/axes/image handles for each cell and then
% closes the figure, so those handles are invalid by the time a caller stores
% or serialises the record. The renderer is frozen and left untouched; only the
% per-cell provenance a caller can legitimately keep is retained here.
if ~isfield(rec,'cells'),return;end
keep={'skipped','skip_reason','result_status','state_kind','base_path','png_path','fig_path'};
c=rec.cells;out=cell(size(c));
for i=1:numel(c)
    s=struct();
    if isstruct(c{i})
        for k=1:numel(keep)
            if isfield(c{i},keep{k}),s.(keep{k})=c{i}.(keep{k});end
        end
    end
    out{i}=s;
end
rec.cells=out;
end

function st=localState(row,cells,ci,binary)
admissible=isfinite(row.k_enter)&&logical(row.hard_gate_pass)&&startsWith(row.status,'PASS');density=[];
if ~isempty(ci)&&isfinite(row.k_enter),density=cells(ci).trajectory.x_post(:,row.k_enter);end
rep='raw';if binary&&~isempty(density),density=ie2a.exact_count_binary(density,.5);rep='exact-count binary';elseif binary,rep='exact-count binary';end
st=struct('density',density,'nelx',row.nelx,'nely',row.nely, ...
    'method',sprintf('%s-%s',row.method,row.method_variant),'status',row.status, ...
    'admissible',admissible,'state_label',sprintf('q=.99, k_{enter}=%g',row.k_enter), ...
    'representation',rep,'domain_extent',[0 8 0 1], ...
    'support_points',[0 .5;8 .5],'load_vectors',zeros(0,4));
end
