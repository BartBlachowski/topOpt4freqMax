function test_detector_precedence()
%TEST_DETECTOR_PRECEDENCE Adversarial health/modal gates at a known true fire.
here=fileparts(mfilename('fullpath')); resultDir=fullfile(here,'results');
s=load(fullfile(resultDir,'development_240x30.mat'),'res'); r=s.res;
j=jsondecode(fileread(fullfile(resultDir,'native_convergence_config.json')));
d=j.selected_detector; k=j.development_result.selected_first_fire;
R=double(r.telemetry.rho_snapshots); n=r.nOuter;
r.telemetry.rho_phase_rms=NaN(1,n); r.telemetry.topology_phase_turnover=NaN(1,n);
for q=2:n
    delta=R(:,q+1)-R(:,q-1);
    r.telemetry.rho_phase_rms(q)=sqrt(mean(delta.^2));
    r.telemetry.topology_phase_turnover(q)=mean((R(:,q+1)>=.5)~=(R(:,q-1)>=.5));
end

caseName={'healthy_stationary';'lp_failure_zero_step';'inner_failure'; ...
    'eigensolver_warning';'nonfinite_state';'stationary_simple_mode'; ...
    'reopened_eigengap';'volume_infeasible'};
expected=[true;false;false;false;false;false;false;false]; actual=false(size(expected));
for z=1:numel(caseName)
    h=r.hist; t=r.telemetry; c=r.cfg; im=k-d.modal_window+1:k;
    switch caseName{z}
        case 'lp_failure_zero_step'
            t.lp_flag(im)=0; t.rho_phase_rms(im)=0; t.topology_phase_turnover(im)=0;
        case 'inner_failure', h.innerConv(im)=false;
        case 'eigensolver_warning', t.eig_warning(im)=true;
        case 'nonfinite_state', t.finite_ok(im)=false;
        case 'stationary_simple_mode', h.N(im)=1;
        case 'reopened_eigengap', t.gaps_rel(1,im)=2*d.gap_tol;
        case 'volume_infeasible', h.vol(im)=c.volfrac*(1+2*d.volume_tol_rel);
    end
    actual(z)=nativeConvergenceDetector(h,t,k,c,d);
end
passed=actual==expected;
tests=table(caseName,expected,actual,passed);
writetable(tests,fullfile(resultDir,'native_convergence_precedence_tests.csv'));
assert(all(passed),'test_detector_precedence:Failed','At least one precedence test failed.');
disp(tests);
end
