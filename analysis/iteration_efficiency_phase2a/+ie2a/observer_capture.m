function observer_capture(rec, k, observer)
%OBSERVER_CAPTURE Persist exactly the density state supplied to history recording.
% Each method stage owns a separate history recorder, so its local k may
% restart. The file counter preserves chronological observation order.
m=matfile(observer.output_file,'Writable',true);
q=m.n_observed+1;
assert(q<=observer.max_states,'ie2a:ObserverCapacity','Observer capacity exceeded.');
x=double(rec.xPhys(:));
assert(numel(x)==observer.n_elements,'ie2a:ObserverDensitySize','Observed density size changed.');
m.xPhys(:,q)=x;
m.iteration(q,1)=double(rec.iter);
m.stage(q,1)=localGet(rec,'stage',1);
m.stage_iteration(q,1)=localGet(rec,'stage_iter',NaN);
m.n_observed=q;
end
function x=localGet(s,name,default)
if isfield(s,name)&&~isempty(s.(name)), x=double(s.(name)); else, x=default; end
end
