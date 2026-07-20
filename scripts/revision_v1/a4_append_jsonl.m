function a4_append_jsonl(path, record)
%A4_APPEND_JSONL  Append one durable telemetry record without rewriting history.
fid = fopen(path, 'a');
if fid < 0, error('a4_append_jsonl:OpenFailed', 'Cannot append %s.', path); end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(record));
end
