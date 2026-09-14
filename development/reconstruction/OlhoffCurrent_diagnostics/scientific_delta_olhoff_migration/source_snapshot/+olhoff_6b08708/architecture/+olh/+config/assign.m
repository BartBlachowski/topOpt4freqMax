function cfg = assign(cfg, varargin)
%ASSIGN  Set several canonical fields at once: assign(cfg,'a.b',1,'c.d',2).
if mod(numel(varargin),2) ~= 0
    error('olh:config:assignPairs','assign expects path/value pairs.');
end
for k = 1:2:numel(varargin)
    cfg = olh.config.setPath(cfg, varargin{k}, varargin{k+1});
end
end
