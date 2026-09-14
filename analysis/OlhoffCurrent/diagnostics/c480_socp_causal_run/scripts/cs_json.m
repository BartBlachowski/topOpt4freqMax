function cs_json(file, s)
%CS_JSON  Write a struct as pretty JSON.
fid = fopen(file, 'w');
assert(fid > 0, 'cs_json:open', 'cannot open %s', file);
fprintf(fid, '%s', jsonencode(s, 'PrettyPrint', true, 'ConvertInfAndNaN', true));
fclose(fid);
end
