function gen_config_reference()
%GEN_CONFIG_REFERENCE  Write CONFIG_REFERENCE.md from olh.config.schema.
%   Generated, never hand-maintained, so it cannot drift from the schema.
root = '/Users/piotrek/Programming/Matlab/Olhoff';
S = olh.config.schema();
f = fopen(fullfile(root,'architecture','docs','CONFIG_REFERENCE.md'),'w');
fprintf(f,'# CONFIG_REFERENCE\n\n');
fprintf(f,'Every field of the canonical configuration.\n\n');
fprintf(f,'**Generated from `olh.config.schema` by `architecture/tests/gen_config_reference.m`.**\n');
fprintf(f,'Do not edit by hand: regenerate it.\n\n');
fprintf(f,'Provenance classes: **A** specified by a Du-Olhoff source; **B** implied by one;\n');
fprintf(f,'**C** under-specified reconstruction choice; **D** later experimental modification.\n');
fprintf(f,'The evidence for each letter is in `SCIENTIFIC_CONFIG_PROVENANCE.md`.\n\n');

cls = S(:,5);
fprintf(f,'| Class | Fields |\n|---|---|\n');
for c = {'A','B','C','D'}
    fprintf(f,'| **%s** | %d |\n', c{1}, sum(strcmp(cls,c{1})));
end
fprintf(f,'| total | %d |\n\n', size(S,1));

% group by top-level branch, preserving schema order
branch = cellfun(@(p) local_head(p), S(:,1), 'UniformOutput', false);
seen = {};
for i = 1:size(S,1)
    if ~any(strcmp(branch{i}, seen))
        seen{end+1} = branch{i}; %#ok<AGROW>
        fprintf(f,'\n## cfg.%s\n\n', branch{i});
        fprintf(f,'| Field | Type | Default | Admissible | Class | Meaning |\n');
        fprintf(f,'|---|---|---|---|---|---|\n');
    end
    fprintf(f,'| `%s` | %s | `%s` | %s | %s | %s |\n', ...
        S{i,1}, S{i,2}, local_val(S{i,3}), local_dom(S{i,2},S{i,4}), S{i,5}, S{i,6});
end
fprintf(f,'\n');
fclose(f);
fprintf('CONFIG_REFERENCE.md written: %d fields\n', size(S,1));
end

function h = local_head(p)
q = strsplit(p,'.');  h = q{1};
end
function s = local_val(v)
if ischar(v)
    s = ['''' v ''''];
elseif isempty(v)
    s = '[]';
elseif islogical(v)
    s = mat2str(v);
elseif isnumeric(v) && isscalar(v)
    s = num2str(v,'%g');
elseif isnumeric(v)
    s = mat2str(v);
else
    s = class(v);
end
end
function s = local_dom(kind, dom)
switch kind
    case 'enum'
        s = strjoin(cellfun(@(x) ['`' x '`'], dom, 'UniformOutput', false), ', ');
    case {'double','int'}
        if isempty(dom)
            s = '-';
        else
            s = sprintf('[%g, %g]', dom(1), dom(2));
        end
    case 'logical', s = '`true`, `false`';
    case 'vector',  s = 'numeric vector or `[]`';
    case 'char',    s = 'text';
    otherwise,      s = 'see validation';
end
end
