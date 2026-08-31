function t = topology_metrics(x, nelx, nely, opts)
%TOPOLOGY_METRICS Frozen exact-count four-neighbour topology gate.
arguments
    x {mustBeNumeric,mustBeReal,mustBeFinite}
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    opts.VolumeFraction (1,1) double = 0.5
    opts.DomainLength (1,1) double = 8
    opts.DomainHeight (1,1) double = 1
    opts.SignificantArea (1,1) double = 0.01
    opts.VolumeRelativeTolerance (1,1) double = 0.001
end
assert(mod(nely,2)==0, 'ie2a:OddNely', 'The frozen midheight-support rule requires even nely.');
assert(numel(x)==nelx*nely, 'ie2a:DensitySize', 'Density field size does not match mesh.');
x = double(x(:));
xb = ie2a.exact_count_binary(x, opts.VolumeFraction);
solid = reshape(logical(xb), nely, nelx);
[labels, sizes] = localComponents(solid);
mid = nely/2;
supportRows = unique([mid, mid+1]);
leftLabels = unique(labels(supportRows,1)); leftLabels(leftLabels==0)=[];
rightLabels = unique(labels(supportRows,end)); rightLabels(rightLabels==0)=[];
spanning = intersect(leftLabels,rightLabels);
requiredConnected = numel(spanning)==1;
if requiredConnected
    requiredLabel = spanning(1);
else
    requiredLabel = 0;
end
detachedLabels = (1:numel(sizes)).';
detachedLabels(detachedLabels==requiredLabel)=[];
detachedSizes = sizes(detachedLabels);
elementArea = opts.DomainLength*opts.DomainHeight/(nelx*nely);
detachedAreas = detachedSizes*elementArea;
strictDetachedPass = isempty(detachedAreas) || all(detachedAreas < opts.SignificantArea);
t = struct();
t.binary = xb;
t.n_solid = nnz(xb);
t.target_n_solid = round(opts.VolumeFraction*nelx*nely);
t.raw_volume_relative_error = abs(mean(x)-opts.VolumeFraction)/opts.VolumeFraction;
t.volume_pass = t.raw_volume_relative_error <= opts.VolumeRelativeTolerance;
t.required_connected = requiredConnected;
t.required_component_label = requiredLabel;
t.component_sizes_elements = sizes;
t.detached_component_sizes_elements = detachedSizes;
t.detached_component_areas = detachedAreas;
t.max_detached_elements = localMaxOrZero(detachedSizes);
t.max_detached_area = localMaxOrZero(detachedAreas);
t.aggregate_detached_elements = sum(detachedSizes);
t.aggregate_detached_area = sum(detachedAreas);
t.n_islands_all = numel(detachedSizes);
t.n_islands_significant = sum(detachedAreas >= opts.SignificantArea);
t.largest_component_fraction = localMaxOrZero(sizes)/max(1,nnz(xb));
t.topology_pass = requiredConnected && strictDetachedPass;
t.hard_gate_pass = t.volume_pass && t.topology_pass;
t.element_area = elementArea;
t.a_sig_elements = opts.SignificantArea/elementArea;
t.aggregate_detached_area_role = 'DIAGNOSTIC_ONLY';
t.n_islands_all_role = 'DIAGNOSTIC_ONLY';
end

function [labels,sizes] = localComponents(solid)
[nr,nc] = size(solid); labels=zeros(nr,nc,'uint32'); sizes=zeros(0,1); id=uint32(0);
queue = zeros(nnz(solid),2); % one allocation, reused by every component
for c=1:nc
    for r=1:nr
        if ~solid(r,c) || labels(r,c)~=0, continue; end
        id=id+1; head=1; tail=1; queue(1,:)=[r c]; labels(r,c)=id; count=0;
        while head<=tail
            rr=queue(head,1); cc=queue(head,2); head=head+1; count=count+1;
            nb=[rr-1 cc;rr+1 cc;rr cc-1;rr cc+1];
            for j=1:4
                rn=nb(j,1); cn=nb(j,2);
                if rn>=1 && rn<=nr && cn>=1 && cn<=nc && solid(rn,cn) && labels(rn,cn)==0
                    tail=tail+1; queue(tail,:)=[rn cn]; labels(rn,cn)=id;
                end
            end
        end
        sizes(double(id),1)=count;
    end
end
end

function y=localMaxOrZero(x)
if isempty(x), y=0; else, y=max(x); end
end
