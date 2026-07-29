addpath('../../Matlab'); addpath('../../../../tools/Matlab');
b=struct('support_type','CC','nelx',40,'nely',5,'volfrac',0.5,'mass_mode','du2007_c1', ...
 'sensitivity_filter',true,'rmin_elem',2.5,'n_target',1,'n_modes',4,'mult_tol',1e-3, ...
 'outer_max_iter',80,'outer_tol',1e-6,'inner_max_iter',30,'inner_tol',1e-4, ...
 'move_lim',0.2,'outer_move',0.2,'alpha',0.5,'acceptance_check',false,'verbose',false);
c1=b; [r1,h1]=topopt_freq_exact(c1);
c2=b; c2.rho_snapshot_interval=1; [r2,h2]=topopt_freq_exact(c2);
fprintf('no-snapshot  final omega1=%.6f\n', h1.final_omega(1));
fprintf('snapshot=1   final omega1=%.6f\n', h2.final_omega(1));
fprintf('omega_trial identical: %d\n', isequal(h1.omega_trial,h2.omega_trial));
fprintf('rho_final    identical: %d   max|diff|=%g\n', isequal(r1,r2), max(abs(r1-r2)));
