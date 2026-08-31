function cleanup = install_observer(outputFile, nElements, maxStates)
%INSTALL_OBSERVER Install an opt-in, write-only post-update density observer.
arguments
    outputFile (1,:) char
    nElements (1,1) double {mustBeInteger,mustBePositive}
    maxStates (1,1) double {mustBeInteger,mustBePositive}
end
p=ie2a.paths(); outputFile=char(java.io.File(outputFile).getCanonicalPath());
allowed={char(java.io.File(p.validation).getCanonicalPath()), ...
    char(java.io.File(p.production).getCanonicalPath())};
assert(any(cellfun(@(d) startsWith(outputFile,[d filesep]),allowed)), ...
    'ie2a:OutputIsolation','Observer output must stay in an isolated Phase-2A output directory.');
folder=fileparts(outputFile); if ~isfolder(folder), mkdir(folder); end
xPhys=nan(nElements,maxStates); %#ok<NASGU>
iteration=nan(maxStates,1); stage=nan(maxStates,1); stage_iteration=nan(maxStates,1); n_observed=0; %#ok<NASGU>
save(outputFile,'xPhys','iteration','stage','stage_iteration','n_observed','-v7.3');
observer=struct('callback',@ie2a.observer_capture,'output_file',outputFile, ...
    'n_elements',nElements,'max_states',maxStates);
key='topopt_iteration_observer_v1';
assert(~isappdata(0,key),'ie2a:ObserverAlreadyInstalled','An iteration observer is already installed.');
setappdata(0,key,observer);
cleanup=onCleanup(@() localRemove(key));
end
function localRemove(key)
if isappdata(0,key), rmappdata(0,key); end
end
