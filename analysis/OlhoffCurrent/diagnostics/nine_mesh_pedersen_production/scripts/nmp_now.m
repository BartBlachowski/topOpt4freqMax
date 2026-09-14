function s = nmp_now()
%NMP_NOW  Local ISO-8601 timestamp with milliseconds and UTC offset.
s = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSXXX'));
end
