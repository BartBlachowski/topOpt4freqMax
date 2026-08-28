function [data, profileId, meta] = performance_benchmark_profile(nelx, nely, mode)
%PERFORMANCE_BENCHMARK_PROFILE  The task struct the performance benchmark runs.
%
%   [data, profileId, meta] = PERFORMANCE_BENCHMARK_PROFILE()
%   [data, profileId, meta] = PERFORMANCE_BENCHMARK_PROFILE(nelx, nely)
%   [data, profileId, meta] = PERFORMANCE_BENCHMARK_PROFILE(nelx, nely, mode)
%
%   MODE selects which interpretation of the benchmark to build:
%
%     'r3'            DEFAULT.  The R3 native-performance profile.  Olhoff runs
%                     to its documented 1600-outer budget.
%
%     'yuksel_table1' DIAGNOSTIC.  Reproduces the reporting interpretation of
%                     Yuksel & Yilmaz (2025) Table 1, in which the eigenvalue-
%                     per-iteration "dynamic code" -- the role our Olhoff column
%                     plays -- is given a FIXED 200-iteration work budget rather
%                     than being run to convergence or to a large budget.
%
%                     Section 6.2 of that paper states it outright: "the
%                     optimization process is terminated after 200 iterations".
%                     Table 1's own numbers agree -- dividing each dynamic-code
%                     total by its per-iteration time gives 8767.8/43.8 = 200.2,
%                     55801.8/279.0 = 200.0 and 208604.1/1043.0 = 200.0.
%
%                     ONLY the outer budget changes.  Nothing about the Olhoff
%                     algorithm is touched: max_outer is the documented
%                     optimization.repro2007 key and reaches OLHOFFOPT as
%                     cfg.maxOuter, the outer for-loop bound and nothing else.
%
%                     The other two methods keep their NATIVE stopping, which is
%                     also what Table 1 does: its "Proposed Method" rows run to
%                     their own stop and report the terminal eigensolve
%                     separately, as the "+ (43.1 s)*" addendum.
%
%   Returns performance_comparison.json decoded and with every override the
%   performance comparison applies on top of it, i.e. exactly the struct handed
%   to RUN_TOPOPT_FROM_JSON.  `data.optimization.approach` is NOT set here --
%   the caller selects the method.
%
%   This function is the SINGLE definition of the benchmark profile.
%   PERFORMANCE_COMPARISON calls it, and so does the Olhoff benchmark-path
%   equivalence harness.  If the two built their own copies, the harness could
%   certify a profile the campaign no longer runs, which is the one failure
%   mode a path-equivalence check must not have.
%
%   OUTPUTS
%     data       decoded task struct with all benchmark overrides applied
%     profileId  identifier recorded in every result artifact
%     meta       provenance: source file, its SHA-256, the overrides applied,
%                and any deviation from BENCHMARK_PROTOCOL_R3.md
%
%   See also PERFORMANCE_COMPARISON, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE,
%            RUN_TOPOPT_FROM_JSON.

here = fileparts(mfilename('fullpath'));
jsonPath = fullfile(here, 'performance_comparison.json');
raw = fileread(jsonPath);
data = jsondecode(raw);

% Normalize spelling to avoid silent "optimisation" vs "optimization" bugs.
if isfield(data, 'optimisation') && ~isfield(data, 'optimization')
    data.optimization = data.optimisation;
end
if ~isfield(data, 'optimization')
    error('performance_benchmark_profile:MissingOptimizationField', ...
        'Missing "optimization" section in %s', jsonPath);
end

% Disable visualization and image saving for clean performance measurement
data.postprocessing.visualize_live    = false;
data.postprocessing.save_final_image  = false;
data.postprocessing.save_snapshot_image = false;

% Fix filter radius to 2 finite elements regardless of resolution.
% The base JSON uses physical units (0.04 m), which gives < 1 element at
% coarser meshes and causes checkerboard patterns.  Switching to 'element'
% units with radius = 2 keeps the filter consistent across all resolutions.
data.optimization.filter.radius       = 2;
data.optimization.filter.radius_units = 'element';

% -------------------------------------------------------------------------
% Du-Olhoff 2007 clean-room reproduction: settings that CANNOT come from the
% shared benchmark block.  See the header comment of PERFORMANCE_COMPARISON
% for why each one is here, and DIAGNOSTIC_REPRO2007_BENCHMARK.md for what
% happened the last time two of them were inherited instead of stated.
data.optimization.repro2007 = struct( ...
    'support_type', 'SS', ...    % bc.supports are closest_point at mid-height
    'move',         0.005, ...   % documented fig3a_best value
    'max_outer',    1600, ...    % documented fig3a_best budget
    'rho_min',      1e-3, ...    % paper eq. (7e); NOT void_material.rho_min
    'tol_outer',    1e-3);       % documented fig3a_best value

if nargin >= 1 && ~isempty(nelx)
    data.domain.mesh.nelx = nelx;
end
if nargin >= 2 && ~isempty(nely)
    data.domain.mesh.nely = nely;
end
if nargin < 3 || isempty(mode)
    mode = 'r3';
end
mode = lower(char(string(mode)));

switch mode
    case 'r3'
        profileId = 'perf_r3_olhoff_du2007repro_fig3a_best_rmin2el';
        modeNote = ['R3 native-performance interpretation: Olhoff runs to its ' ...
                    'documented 1600-outer budget.'];
        yukselDeviations = {};
    case 'yuksel_table1'
        data.optimization.repro2007.max_outer = 200;
        profileId = 'diag_yukseltable1_olhoff_du2007repro_fig3a_best_rmin2el_200outer';
        modeNote = ['Yuksel & Yilmaz (2025) Table 1 reporting interpretation: the ' ...
                    'eigenvalue-per-iteration method is given a FIXED 200-iteration ' ...
                    'work budget (paper section 6.2, "terminated after 200 ' ...
                    'iterations"); the other methods keep their native stopping.'];
        % Stated, not hidden.  Section 6.2 also specifies three OTHER settings for
        % its dynamic code.  They are deliberately NOT adopted here: this mode
        % changes the outer budget only, so that its Olhoff column remains the
        % same operating point as the R3 column and the two are comparable.
        % Adopting them would make this a reproduction of Yuksel's dynamic CODE
        % rather than of Table 1's reporting interpretation, and would need its
        % own equivalence proof and its own justification.
        yukselDeviations = { ...
            ['move limit: this profile keeps 0.005 (documented fig3a_best); ' ...
             'Yuksel section 6.2 reduces its dynamic code to 0.01 after finding ' ...
             '0.2 caused convergence issues']
            ['filter radius: this profile keeps r_min = 2.0 elements; Yuksel ' ...
             'section 6.2 uses a sensitivity filter radius of 2.5']
            ['multiplicity tolerance: this profile keeps tolMult = 0.05; Yuksel ' ...
             'section 6.2 treats omega_1 and omega_2 as repeated below a 4% ' ...
             'difference, i.e. 0.04']};
    otherwise
        error('performance_benchmark_profile:UnknownMode', ...
            ['Unknown mode "%s".  Available: ''r3'', ''yuksel_table1''.'], mode);
end

meta = struct();
meta.profile_id            = profileId;
meta.mode                  = mode;
meta.mode_note             = modeNote;
meta.protocol_profile_id   = 'olhoff_du_2007_repro_fig3a_best';
meta.protocol_document     = 'BENCHMARK_PROTOCOL_R3.md';
meta.source_json           = 'examples/Performance/performance_comparison.json';
meta.source_json_sha256    = sha256_hex(raw);
meta.overrides_applied     = { ...
    'postprocessing.visualize_live = false'
    'postprocessing.save_final_image = false'
    'postprocessing.save_snapshot_image = false'
    'optimization.filter.radius = 2'
    'optimization.filter.radius_units = ''element'''
    'optimization.repro2007.support_type = ''SS'''
    'optimization.repro2007.move = 0.005'
    'optimization.repro2007.max_outer = 1600'
    'optimization.repro2007.rho_min = 1e-3'
    'optimization.repro2007.tol_outer = 1e-3'};
if strcmp(mode, 'yuksel_table1')
    meta.overrides_applied{end+1} = 'optimization.repro2007.max_outer = 200  (mode: yuksel_table1)';
    meta.yuksel_table1 = struct( ...
        'source', 'Yuksel & Yilmaz (2025), Engineering Computations 42(9), Table 1 and section 6.2', ...
        'file', 'references/Yuksel2025_Efficient.pdf', ...
        'stated_rule', 'the optimization process is terminated after 200 iterations', ...
        'implied_counts_from_table1', ['8767.8/43.8 = 200.2 (160x20); ' ...
            '55801.8/279.0 = 200.0 (240x30); 208604.1/1043.0 = 200.0 (320x40)'], ...
        'deviations_not_adopted', {yukselDeviations});
end

% Stated, not hidden: the executed filter radius is the benchmark's shared
% cross-resolution setting, not the radius the R3 protocol records for this
% method's native profile.  The two are different operating points of the same
% configuration lineage, and the difference belongs in every artifact that
% cites this profile.
meta.deviations_from_protocol = { ...
    ['filter radius: executed r_min = 2.0 elements (shared benchmark ' ...
     'cross-resolution setting); BENCHMARK_PROTOCOL_R3.md section 3.4 ' ...
     'records r_min = 1.3 elements for profile_id ' ...
     'olhoff_du_2007_repro_fig3a_best.  r_min = 1.3 is the radius that ' ...
     'reproduces paper Fig. 3a; r_min = 2.0 is a valid operating point of ' ...
     'the same method but is NOT the paper-reproduction figure.']};
end
