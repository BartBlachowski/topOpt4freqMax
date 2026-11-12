function T = sec2_3_spectral_check()
%SEC2_3_SPECTRAL_CHECK  Section 2.3: the spectral form of the compliance on the
%   single-load designs (alpha = 1) of the three examples, about 3 min.
%
%   For every iterate: modal mass m_v = v'M(x)v of the reference shape v = phi_1^(0),
%   lambda_hat = s^2 m_v / c (Eq. (7), optimization model) and the lowest structural
%   omega_1 (common evaluator).  For the final design: MAC_M between v and the lowest
%   structural mode (w_ks of Eq. (8)) and sqrt(lambda_hat) versus omega_1.
%   Output: results/spectral/spectral.csv and spectral_history.png.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('spectral');
probs = {problem_ss_beam(), problem_clamped_beam(), problem_building()};
rows = {};  H = {};
for ip = 1:numel(probs)
    prob = probs{ip};
    P = proposed_freq(prob, struct('keepDesigns', true));
    mdl = P.mdl;  f = mdl.free;  v = P.phi0(:,1);  s = P.omega0(1)^2;
    n = numel(P.designs);  mv = zeros(n,1);  lh = zeros(n,1);  om = zeros(n,1);
    for k = 1:n
        x = P.designs{k};
        [K, M] = assemble_KM(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin);
        F = s*(M*v);  u = zeros(mdl.ndof,1);  u(f) = K(f,f)\F(f);
        mv(k) = v(f)'*M(f,f)*v(f);  lh(k) = s^2*mv(k)/(F(f)'*u(f));
        om(k) = structural_modes(prob, x, 1, struct('mdl', mdl));
    end
    [~, Phi] = structural_modes(prob, P.x, 1, struct('mdl', mdl, 'Emin', prob.Emin, 'rhoMin', prob.rhoMin));
    [~, M] = assemble_KM(mdl, P.x, prob.E, prob.Emin, prob.rho, prob.rhoMin);
    wks = mac_mass(v(f), Phi(f,1), M(f,f));
    fprintf(['%-13s m_v %.4f -> %.4f (%+.1f%%) | final: MAC_M %.4f, sqrt(lambda_hat) %.2f vs omega1 %.2f (%+.2f%%) | ' ...
        'last ten iterates max |sqrt(lh)-omega1|/omega1 = %.2f%%\n'], prob.name, mv(1), mv(end), 100*(mv(end)/mv(1)-1), ...
        wks, sqrt(lh(end)), om(end), 100*(sqrt(lh(end))/om(end)-1), 100*max(abs(sqrt(lh(end-9:end)) - om(end-9:end))./om(end-9:end)));
    rows(end+1, :) = {prob.name, mv(1), mv(end), wks, sqrt(lh(end)), om(end)}; %#ok<AGROW>
    H{ip} = [sqrt(lh), om, mv]; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'case', 'm_v_start', 'm_v_final', 'MAC_M', 'sqrt_lambda_hat', 'omega1'});
writetable(T, fullfile(out, 'spectral.csv'));
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 350]);
for ip = 1:3
    subplot(1,3,ip);  yyaxis left;  plot(H{ip}(:,1:2), 'LineWidth', 1.2);  ylabel('rad/s');
    yyaxis right;  plot(H{ip}(:,3), ':', 'LineWidth', 1.2);  ylabel('m_v');  grid on;
    title(probs{ip}.name, 'Interpreter', 'none');  xlabel('iteration');
end
exportgraphics(fig, fullfile(out, 'spectral_history.png'), 'Resolution', 200);  close(fig);
end
