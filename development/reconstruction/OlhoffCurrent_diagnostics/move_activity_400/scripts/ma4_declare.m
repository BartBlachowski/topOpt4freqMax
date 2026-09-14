function ma4_declare()
%MA4_DECLARE  Register this study's raw trajectories as REQUIRED evidence.
%
%   This is the step the three earlier studies never had.  After it runs,
%   olhoffcurrent_evidence_gate FAILS this study if either trajectory is ever
%   deleted or altered -- which is exactly what silently happened before.
%
%   olhoffcurrent_evidence_declare hashes the files on disk and REFUSES to
%   declare a required artifact that is not there, so this cannot succeed
%   unless the evidence genuinely exists.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = olhoffcurrent_root();

items = { 'P400_400x50_trajectory.mat', 'required', ...
            ['ARM P400 element-level trajectory: RHO and DRHO (NE x nOuter, double), ' ...
             'move, full hist, resolved cfg.  Production move ladder.'] ; ...
          'F400_400x50_trajectory.mat', 'required', ...
            ['ARM F400 element-level trajectory: RHO and DRHO (NE x nOuter, double), ' ...
             'move, full hist, resolved cfg.  Fixed move 0.04 counterfactual.'] };

extra = struct();
extra.preregistration_sha256 = '706b8865075f97cb0d5824658fa6e562636d464dba2611580d6c10fdff7716d4';
extra.mesh = [400 50];
extra.NE = 20000;
extra.reconstructs = {'rho(k)','rho(k-1)','Delta rho_e(k)','move(k)'};
extra.convention = ['u_e(k) = |rho_e(k)-rho_e(k-1)| / move(k); move(k) bounds the ' ...
                    'increment stored at index k (olhoffSolve l.267/298/373/381)'];

olhoffcurrent_evidence_declare(study, 'move_activity_400', items, ...
    'EvidenceRoot', 'analysis/OlhoffCurrent/evidence/move_activity_400', 'Extra', extra);

st = olhoffcurrent_evidence_gate(study, 'Verbose', true);
assert(st.ok, 'ma4_declare:GateFailed', 'evidence gate FAILED: %s', st.detail);
fprintf('[ma4_declare] evidence gate PASS -- %s\n', st.detail);
end
