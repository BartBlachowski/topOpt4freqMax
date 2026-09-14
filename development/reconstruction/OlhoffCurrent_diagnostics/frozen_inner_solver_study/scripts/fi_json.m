function fi_json(path,value)
f=fopen(path,'w'); assert(f>=0); c=onCleanup(@() fclose(f));
fprintf(f,'%s\n',jsonencode(value,'PrettyPrint',true));
end
