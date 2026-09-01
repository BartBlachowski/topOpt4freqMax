function audit_all_stationarity(tags)
%AUDIT_ALL_STATIONARITY  Run the WP2 certificate over every named result dir.
for i=1:numel(tags)
    t=tags{i};
    try
        audit_stationarity(t);
    catch ME
        fprintf('\n***** WP2 AUDIT FAILED for %s: %s\n%s\n',t,ME.identifier,ME.message);
    end
end
end
