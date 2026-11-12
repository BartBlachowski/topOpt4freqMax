function compare_with_paper(label, computed, paper, unit)
%COMPARE_WITH_PAPER  One line: computed value, value printed in the paper, difference.
if nargin < 4, unit = ''; end
fprintf('  %-46s %10.4g  (paper %10.4g %s, %+6.2f%%)\n', label, computed, paper, unit, ...
    100*(computed - paper)/abs(paper));
end
