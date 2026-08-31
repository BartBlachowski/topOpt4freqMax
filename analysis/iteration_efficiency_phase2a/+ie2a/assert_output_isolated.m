function canonical = assert_output_isolated(outputDir, purpose)
%ASSERT_OUTPUT_ISOLATED Prevent writes into frozen/audit/performance artifacts.
arguments
    outputDir (1,:) char
    purpose (1,:) char {mustBeMember(purpose,{'validation','production'})}
end
p=ie2a.paths(); canonical=char(java.io.File(outputDir).getCanonicalPath());
if strcmp(purpose,'validation'),base=p.validation;else,base=p.production;end
base=char(java.io.File(base).getCanonicalPath());
assert(strcmp(canonical,base)||startsWith(canonical,[base filesep]),'ie2a:OutputIsolation', ...
    'Output must remain under %s.',base);
end
