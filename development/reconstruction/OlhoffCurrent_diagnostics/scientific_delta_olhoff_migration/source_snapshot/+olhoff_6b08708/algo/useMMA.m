function resolved = useMMA(variant)
%USEMMA  Select which MMA copy resolves on the path (see mma_published/README).
%
%   'published' -- Svanberg's published September-2007 constants
%                  (move = 0.5, asyinit = 0.5).  Reproduction BASELINE.
%   'asfound'   -- the copy as received in mma/ (move = 1.0, asyinit = 0.01),
%                  a local modification inherited from the user's lineage.
%
%   Returns the resolved full path of mmasub.m, to be recorded in the run.

root   = fileparts(fileparts(mfilename('fullpath')));
pubDir = fullfile(root, 'mma_published');
asDir  = fullfile(root, 'mma');

onPath = @(d) any(strcmp(d, strsplit(path, pathsep)));

switch lower(variant)
    case 'published'
        if onPath(asDir),   rmpath(asDir);   end
        if ~onPath(pubDir), addpath(pubDir, '-begin'); end
    case 'asfound'
        if onPath(pubDir),  rmpath(pubDir);  end
        if ~onPath(asDir),  addpath(asDir, '-begin');  end
    otherwise
        error('useMMA:variant', 'unknown mmaVariant %s', variant);
end

resolved = which('mmasub');
expect = struct('published', pubDir, 'asfound', asDir);
if ~strncmp(resolved, expect.(lower(variant)), numel(expect.(lower(variant))))
    error('useMMA:resolve', 'mmasub resolves to %s, expected under %s', ...
          resolved, expect.(lower(variant)));
end
end
