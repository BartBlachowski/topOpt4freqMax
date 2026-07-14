function out = a4_mode_screen(Phi, omegas, xPhys, ctx, phiPrev)
%A4_MODE_SCREEN  Mode-admissibility screen and selection for A4 (spec V3 §4.3.1).
%
%   out = A4_MODE_SCREEN(Phi, omegas, xPhys, ctx)
%   out = A4_MODE_SCREEN(Phi, omegas, xPhys, ctx, phiPrev)
%
%   Implements the support-connectivity screen defined in
%   examples/Revision_v1/A4_SPECIFICATION_V3.md §4.3.1.  This is the SINGLE
%   implementation of the rule; it is reused by
%     - topopt_freq.m            (R-1 refresh: mode selection at each refresh)
%     - a4_preflight_spectral_screen.m (Gate A4-Pre)
%     - a4_eigenpair_refresh.m   (endpoint diagnosis)
%     - check_a4_run.m           (B3 classification)
%
%   WHY CONNECTIVITY AND NOT VOID KINETIC ENERGY
%   --------------------------------------------
%   The screen was originally specified on the kinetic-energy fraction in
%   low-density elements.  That quantity is 0.0000 for every observed mode at
%   every pmass tested, so such a screen would never fire.  The spurious modes
%   in this repository are solid components NOT connected to the supports.  The
%   screen therefore tests connectivity.  See spec §0.2 and §4.3.1.
%
%   ADMISSIBILITY (all must hold; thresholds are spec-declared, not tuned)
%     largest_support_component_kinetic_fraction >= 0.5
%     dominant_component_touches_both_supports   == true
%     low_density_strain_fraction                <= 0.5
%     mac_prev                                   >= 0.8   (only if phiPrev given)
%
%   ctx fields (all required):
%     nelx, nely, edofMat, KE, ME, M, free
%     Emax, Emin, rho0, rho_min, penal, massInterp
%   ctx optional:
%     lowDensityThreshold (default 0.05), solidThreshold (default 0.5)
%     macThreshold (default 0.8), supportKineticThreshold (default 0.5)
%     lowDensityStrainThreshold (default 0.5)
%
%   out.modes(k)   per-mode metrics + .admissible
%   out.selected   index of the selected mode (0 if none admissible)
%   out.reason     human-readable selection/rejection reason
%   out.nComponents            number of solid components
%   out.largestSupportCompId   id of the largest support-connected component

if nargin < 5, phiPrev = []; end

lowThr      = localField(ctx, 'lowDensityThreshold', 0.05);
solidThr    = localField(ctx, 'solidThreshold', 0.5);
macThr      = localField(ctx, 'macThreshold', 0.8);
supKinThr   = localField(ctx, 'supportKineticThreshold', 0.5);
lowStrainThr= localField(ctx, 'lowDensityStrainThreshold', 0.5);

x = xPhys(:);
nModes = numel(omegas);

% ---- design-dependent quantities (computed once, not per mode) -----------
[component, compStats] = localSolidComponents(x, ctx.nelx, ctx.nely, solidThr);
largestCompId = localLargestSupportComponent(compStats);
lowMask = x < lowThr;

out = struct();
out.nComponents = numel(compStats);
out.largestSupportCompId = largestCompId;
out.lowDensityThreshold = lowThr;
out.solidThreshold = solidThr;
out.modes = repmat(localBlankMode(), 0, 1);

% ---- per-mode metrics ----------------------------------------------------
for k = 1:nModes
    row = localBlankMode();
    row.index = k;
    row.omega = omegas(k);

    if ~isfinite(omegas(k))
        row.admissible = false;
        row.reject_reason = 'eigenvalue not finite';
        out.modes(end+1, 1) = row; %#ok<AGROW>
        continue;
    end

    phi = Phi(:, k);
    [kin, str] = localElementEnergies(phi, omegas(k), x, ctx);

    row.kinetic_energy_total = sum(kin);
    row.strain_energy_total  = sum(str);
    row.low_density_kinetic_fraction = localFrac(kin(lowMask), kin);
    row.low_density_strain_fraction  = localFrac(str(lowMask), str);

    [domId, domK, domS] = localDominantComponent(component, kin, str);
    row.dominant_solid_component_id = domId;
    row.dominant_solid_component_kinetic_fraction = domK;
    row.dominant_solid_component_strain_fraction  = domS;
    if domId > 0 && domId <= numel(compStats)
        row.dominant_component_touches_both_supports = ...
            compStats(domId).touches_left && compStats(domId).touches_right;
    end
    if largestCompId > 0
        supportMask = component == largestCompId;
        row.largest_support_component_kinetic_fraction = localFrac(kin(supportMask), kin);
    end

    if ~isempty(phiPrev)
        row.mac_prev = a4_mac(phi, phiPrev, ctx.M);
    end

    % ---- admissibility (spec §4.3.1) ------------------------------------
    reasons = {};
    if ~(row.largest_support_component_kinetic_fraction >= supKinThr)
        reasons{end+1} = sprintf('support-connected kinetic fraction %.4f < %.2f', ...
            row.largest_support_component_kinetic_fraction, supKinThr); %#ok<AGROW>
    end
    if ~row.dominant_component_touches_both_supports
        reasons{end+1} = 'dominant component does not touch both supports'; %#ok<AGROW>
    end
    if ~(row.low_density_strain_fraction <= lowStrainThr)
        reasons{end+1} = sprintf('low-density strain fraction %.4f > %.2f', ...
            row.low_density_strain_fraction, lowStrainThr); %#ok<AGROW>
    end
    if ~isempty(phiPrev) && ~(row.mac_prev >= macThr)
        reasons{end+1} = sprintf('MAC to previous reference %.4f < %.2f', ...
            row.mac_prev, macThr); %#ok<AGROW>
    end

    row.admissible = isempty(reasons);
    if ~row.admissible
        row.reject_reason = strjoin(reasons, '; ');
    end
    out.modes(end+1, 1) = row; %#ok<AGROW>
end

% ---- selection: MAC continuity among admissible modes --------------------
% Spec: "the mode is selected by MAC continuity against the previously used
% Phi, among modes passing the localization screen -- never by raw index."
adm = find([out.modes.admissible]);
if isempty(adm)
    out.selected = 0;
    out.reason = 'no admissible mode: every candidate failed the support-connectivity screen';
    return;
end

if isempty(phiPrev)
    % No continuity reference (first screen): lowest admissible mode.
    out.selected = adm(1);
    out.reason = sprintf('lowest admissible mode (index %d); no continuity reference supplied', adm(1));
else
    macs = [out.modes(adm).mac_prev];
    [bestMac, ii] = max(macs);
    out.selected = adm(ii);
    out.reason = sprintf('max MAC continuity %.4f at index %d among %d admissible mode(s)', ...
        bestMac, out.selected, numel(adm));
end
end

% =========================================================================

function v = localField(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
end
end

function row = localBlankMode()
row = struct( ...
    'index', 0, ...
    'omega', NaN, ...
    'kinetic_energy_total', NaN, ...
    'strain_energy_total', NaN, ...
    'low_density_kinetic_fraction', NaN, ...
    'low_density_strain_fraction', NaN, ...
    'dominant_solid_component_id', 0, ...
    'dominant_solid_component_kinetic_fraction', NaN, ...
    'dominant_solid_component_strain_fraction', NaN, ...
    'dominant_component_touches_both_supports', false, ...
    'largest_support_component_kinetic_fraction', NaN, ...
    'mac_prev', NaN, ...
    'admissible', false, ...
    'reject_reason', '');
end

function [kineticElem, strainElem] = localElementEnergies(phi, omega, x, ctx)
% Vectorized form of the S1 diagnostic element energies (same formulas).
%   strain_e  = 0.5 * [Emin + x^p (E0-Emin)]      * ue' KE ue
%   kinetic_e = 0.5 * omega^2 * rho_e             * ue' ME ue
Ue = phi(ctx.edofMat);                 % nEl x 8
if ~isreal(Ue), Ue = real(Ue); end
keScale  = ctx.Emin + x.^ctx.penal * (ctx.Emax - ctx.Emin);
[rhoScale, ~] = our_mass_interpolation(x, ctx.rho0, ctx.rho_min, ...
    ctx.massInterp.mode, ctx.massInterp.pmass);
rhoScale = rhoScale(:);

strainElem  = 0.5 * keScale(:) .* sum((Ue * ctx.KE) .* Ue, 2);
kineticElem = 0.5 * omega^2 * rhoScale .* sum((Ue * ctx.ME) .* Ue, 2);
end

function f = localFrac(partVals, allVals)
den = sum(max(real(allVals(:)), 0));
if den <= 0
    f = NaN;
else
    f = sum(max(real(partVals(:)), 0)) / den;
end
end

function [component, stats] = localSolidComponents(x, nelx, nely, threshold)
solid = reshape(x(:) >= threshold, nely, nelx);
component = zeros(nely, nelx);
stats = repmat(struct('id', 0, 'size', 0, 'touches_left', false, 'touches_right', false), 0, 1);
compId = 0;
nSolid = nnz(solid);
for ix = 1:nelx
    for iy = 1:nely
        if ~solid(iy, ix) || component(iy, ix) ~= 0
            continue;
        end
        compId = compId + 1;
        q = zeros(nSolid, 2);
        head = 1; tail = 1;
        q(tail, :) = [iy, ix];
        component(iy, ix) = compId;
        st = struct('id', compId, 'size', 0, 'touches_left', false, 'touches_right', false);
        while head <= tail
            cy = q(head, 1); cx = q(head, 2);
            head = head + 1;
            st.size = st.size + 1;
            st.touches_left  = st.touches_left  || cx == 1;
            st.touches_right = st.touches_right || cx == nelx;
            neigh = [cy-1, cx; cy+1, cx; cy, cx-1; cy, cx+1];
            for a = 1:4
                ny = neigh(a, 1); nx = neigh(a, 2);
                if ny >= 1 && ny <= nely && nx >= 1 && nx <= nelx && ...
                        solid(ny, nx) && component(ny, nx) == 0
                    tail = tail + 1;
                    q(tail, :) = [ny, nx];
                    component(ny, nx) = compId;
                end
            end
        end
        stats(end+1, 1) = st; %#ok<AGROW>
    end
end
component = component(:);
end

function id = localLargestSupportComponent(stats)
id = 0; best = -Inf;
for i = 1:numel(stats)
    if stats(i).touches_left && stats(i).touches_right && stats(i).size > best
        best = stats(i).size;
        id = stats(i).id;
    end
end
end

function [domId, fracK, fracS] = localDominantComponent(component, kineticElem, strainElem)
ids = unique(component(component > 0));
if isempty(ids)
    domId = 0; fracK = NaN; fracS = NaN;
    return;
end
bestK = -Inf; domId = 0;
for i = 1:numel(ids)
    mask = component == ids(i);
    kFrac = localFrac(kineticElem(mask), kineticElem);
    if kFrac > bestK
        bestK = kFrac;
        domId = ids(i);
    end
end
mask = component == domId;
fracK = localFrac(kineticElem(mask), kineticElem);
fracS = localFrac(strainElem(mask), strainElem);
end
