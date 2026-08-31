function plan = timing_replay_plan(methods, horizons)
%TIMING_REPLAY_PLAN Balanced serial one-thread schedule after endpoint freeze.
arguments
    methods (:,1) string
    horizons (:,1) double {mustBeInteger,mustBePositive}
end
assert(numel(methods)==numel(horizons),'ie2a:TimingPlan','One frozen horizon is required per method.');
n=numel(methods); rows=cell(n*(1+3),5); q=0;
for i=1:n, q=q+1; rows(q,:)={methods(i),horizons(i),0,true,i}; end
for rep=1:3
    order=circshift((1:n).',-(rep-1));
    for pos=1:n, i=order(pos);q=q+1;rows(q,:)={methods(i),horizons(i),rep,false,pos};end
end
plan=cell2table(rows,'VariableNames',{'method','horizon','repetition','discarded_warmup','order_position'});
plan.threads=ones(height(plan),1); plan.serial=true(height(plan),1);
end
