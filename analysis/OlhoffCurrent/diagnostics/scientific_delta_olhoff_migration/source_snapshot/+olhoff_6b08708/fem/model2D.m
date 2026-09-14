function mdl = model2D(cfg)
%MODEL2D  Build the 2D beam model of Du & Olhoff (2007), Figs. 2(a-c).
%
%   Domain a x b, nelx x nely rectangular elements, plane stress, thickness t.
%   Node numbering follows top88: nodenrs is (nely+1) x (nelx+1) with row 1 at
%   the TOP of the beam and column 1 at the LEFT end.  dofs are [ux uy] per node.
%
%   cfg fields (all required unless noted):
%     nelx, nely, a, b, t, E, nu, rhom
%     bc        'a' simply supported ends
%               'b' left end clamped, right end simply supported
%               'c' clamped ends
%     support   idealization of a SIMPLE support -- UNRESOLVED in the paper:
%               'mid'    single node at mid-height of the end face
%               'corner' single node at the bottom corner of the end face
%               'face'   whole end face restrained in y (and x at the pinned end)
%     axial     which ends carry ux restraint (simple-support cases only):
%               'one'  pin + roller  (classical, axially free)
%               'both' ux fixed at both ends (axially restrained)
%     elemType  'Q4' | 'Q6'          massType 'consistent' | 'lumped'
%
%   Returns a struct carrying mesh, connectivity, free dofs, the solid element
%   matrices and the assembly index vectors.

if ~isfield(cfg,'axial') || isempty(cfg.axial), cfg.axial = 'one'; end

nelx = cfg.nelx; nely = cfg.nely;
dx = cfg.a/nelx;  dy = cfg.b/nely;

nele  = nelx*nely;
nnode = (nelx+1)*(nely+1);
ndof  = 2*nnode;

% ---- connectivity (top88 convention) ------------------------------------
nodenrs = reshape(1:nnode, 1+nely, 1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1, nele, 1);
edofMat = repmat(edofVec,1,8) + repmat([0 1 2*nely+[2 3 0 1] -2 -1], nele, 1);

iK = reshape(kron(edofMat,ones(8,1))',64*nele,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nele,1);

% ---- element centroid coordinates (column-major: e = (elx-1)*nely+ely) ---
[ely, elx] = ndgrid(1:nely, 1:nelx);
cx = (elx(:)-0.5)*dx;
cy = (ely(:)-0.5)*dy;          % measured DOWNWARD from the top edge

% ---- solid element matrices ---------------------------------------------
[K0, M0] = elemMats2D(dx, dy, cfg.E, cfg.nu, cfg.rhom, cfg.t, ...
                      cfg.elemType, cfg.massType);

% ---- boundary conditions -------------------------------------------------
switch lower(cfg.support)
    case 'mid'
        if mod(nely,2) ~= 0
            error('model2D:mid','support=''mid'' needs an even nely (got %d)',nely);
        end
        rowSS = nely/2 + 1;
    case 'corner'
        rowSS = nely + 1;                  % bottom edge
    case 'face'
        rowSS = [];                        % handled below
    otherwise
        error('model2D:support','unknown support %s',cfg.support);
end

leftCol = 1;  rightCol = nelx+1;
fixed = [];

    function d = dofsOf(nodes, comp)       % comp: 1=ux, 2=uy
        d = 2*nodes(:) - 2 + comp;
    end

clampFace = @(col) reshape([dofsOf(nodenrs(:,col),1); dofsOf(nodenrs(:,col),2)],[],1);

if strcmpi(cfg.support,'face')
    ssY = @(col) dofsOf(nodenrs(:,col),2);          % whole face restrained in y
    ssX = @(col) dofsOf(nodenrs(:,col),1);
else
    ssY = @(col) dofsOf(nodenrs(rowSS,col),2);      % single node
    ssX = @(col) dofsOf(nodenrs(rowSS,col),1);
end

switch lower(cfg.bc)
    case 'a'                                  % simply supported both ends
        fixed = [ssY(leftCol); ssY(rightCol); ssX(leftCol)];
        if strcmpi(cfg.axial,'both')
            fixed = [fixed; ssX(rightCol)];
        end
    case 'b'                                  % left clamped, right simple
        fixed = [clampFace(leftCol); ssY(rightCol)];
        if strcmpi(cfg.axial,'both')
            fixed = [fixed; ssX(rightCol)];
        end
    case 'c'                                  % both ends clamped
        fixed = [clampFace(leftCol); clampFace(rightCol)];
    otherwise
        error('model2D:bc','unknown bc %s',cfg.bc);
end

fixed = unique(fixed(:));
free  = setdiff((1:ndof)', fixed);

mdl = struct('cfg',cfg,'nelx',nelx,'nely',nely,'dx',dx,'dy',dy, ...
             'nele',nele,'nnode',nnode,'ndof',ndof, ...
             'nodenrs',nodenrs,'edofMat',edofMat,'iK',iK,'jK',jK, ...
             'K0',K0,'M0',M0,'free',free,'fixed',fixed, ...
             'cx',cx,'cy',cy,'Ve',dx*dy*cfg.t);
end
