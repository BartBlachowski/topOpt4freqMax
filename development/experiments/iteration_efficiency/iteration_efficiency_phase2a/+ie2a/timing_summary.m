function T = timing_summary(samples)
%TIMING_SUMMARY Median/range/MAD for non-warmup fixed-horizon replay samples.
required={'method','component','seconds','discarded_warmup'};
assert(all(ismember(required,samples.Properties.VariableNames)),'ie2a:TimingSamples','Timing sample schema is incomplete.');
s=samples(~samples.discarded_warmup,:); groups=findgroups(s.method,s.component);
method=splitapply(@(x)x(1),s.method,groups);component=splitapply(@(x)x(1),s.component,groups);
median_seconds=splitapply(@median,s.seconds,groups);min_seconds=splitapply(@min,s.seconds,groups);
max_seconds=splitapply(@max,s.seconds,groups);mad_seconds=splitapply(@(x)median(abs(x-median(x))),s.seconds,groups);
n=splitapply(@numel,s.seconds,groups);
T=table(method,component,median_seconds,min_seconds,max_seconds,mad_seconds,n);
end
