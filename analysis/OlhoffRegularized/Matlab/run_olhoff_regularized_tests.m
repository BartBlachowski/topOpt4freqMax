function run_olhoff_regularized_tests()
%RUN_OLHOFF_REGULARIZED_TESTS Focused route and accounting smoke tests.
routes={"olhoff","lp";"olhoff","mma";"ks","lp";"ks","mma"};
for i=1:size(routes,1)
    formulation=routes{i,1};optimizer=routes{i,2};
    rc=struct('verbose',false,'formulation',formulation,'optimizer',optimizer, ...
        'max_outer_iterations',2,'max_inner_iterations',150,'min_inner',5, ...
        'max_trial_steps',3,'persistence',2);
    [rho,w,info]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rc);
    assert(all(isfinite(rho))&&all(isfinite(w)),'Nonfinite route result.');
    assert(abs(mean(rho)-.5)<1e-4,'Volume constraint drifted.');
    assert(info.cfg.maxOuter==2&&info.cfg.maxInner==150,'Iteration limits were not authoritative.');
    assert(info.iterations.outer<=2&&info.iterations.trial_total<=6,'Iteration accounting exceeded its cap.');
    assert(strcmp(info.formulation,formulation)&&strcmp(info.optimizer,optimizer),'Route metadata mismatch.');
    assert(info.iterations.accepted_updates>=1,'Route %s/%s accepted no smoke update.',formulation,optimizer);
end

rc=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',1,'max_inner_iterations',10,'max_trial_steps',2);
[rho,w,info]=topopt_olhoff_regularized(8,4,.5,3,1.3,.005,'cantilever',rc);
assert(info.model.tipMassValue>0&&all(isfinite(rho))&&all(isfinite(w)),'Cantilever point-mass smoke failed.');
fprintf('OLHOFF_REGULARIZED_TESTS_PASS routes=4 cantilever=1\n');
end
