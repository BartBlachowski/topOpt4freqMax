function [S,P,L,R,E] = fi_setup()
here=fileparts(mfilename('fullpath')); study=fileparts(here);
addpath(fullfile(fileparts(study),'frozen_problem25_reference','scripts'));
S=fp_setup();
E=fullfile(study,'evaluations');
L=load(fullfile(S.study,'evaluations','frozen_ctx.mat'));
R=load(fullfile(S.study,'evaluations','conic_reference.mat'));
P=fp_problem(L.ctx);
assert(isequal(P.rho,S.rho385));
assert(strcmp(S.g('optimizer.inner.variant'),'published'));
end
