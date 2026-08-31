function iteration_efficiency_campaign
%ITERATION_EFFICIENCY_CAMPAIGN Manual production entry point (intentionally locked).
% After pre-production review, paste the review-issued literal token below.
authorizationToken = '';
olhoffVariant = 'lp'; % 'lp' principal, 'mma' secondary, or 'both' as separate rows

addpath(fileparts(mfilename('fullpath')));
p=ie2a.paths();
addpath(fullfile(p.repo,'tools','Matlab'));
addpath(fullfile(p.repo,'analysis','three_method_parametric_study'));
ie2a.production_preflight(RequireAuthorization=true, ...
    AuthorizationToken=authorizationToken,SelectedOlhoffVariant=olhoffVariant);

ie2a.run_production_campaign(p.production,OlhoffVariant=olhoffVariant);
end
