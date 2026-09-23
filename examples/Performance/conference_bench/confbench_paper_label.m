function name = confbench_paper_label(methodKey)
%CONFBENCH_PAPER_LABEL  The label a method carries in paper-facing figures.
%
%   Figures shown to the reader name each method by its authors only
%   ("Du-Olhoff", "Yuksel-Yilmaz") or as "Proposed".  Implementation details
%   (reconstruction, material law, controller) are internal and stay out of
%   the figures.
%
%   This is NOT the record identity.  Records, CSV tables, metadata and the
%   cross-check against the scaling table keep CONFBENCH_DISPLAY_NAME, which
%   distinguishes the Du-Olhoff realizations from one another.
switch lower(char(string(methodKey)))
    case 'olhoff';                    name = 'Du-Olhoff';
    case 'yuksel';                    name = 'Yuksel-Yilmaz';
    case {'proposed','ourapproach'};  name = 'Proposed';
    otherwise
        error('confbench_paper_label:UnknownMethod', 'Unknown method "%s".', methodKey);
end
end
