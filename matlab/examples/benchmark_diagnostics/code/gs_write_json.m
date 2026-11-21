function gs_write_json(path,value)
%GS_WRITE_JSON Write auditable status/metadata; nonfinite JSON numbers are null.
f = fopen(path,'w');
if f < 0, error('GridShell:Output','Cannot write %s',path); end
cleanup = onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value,'PrettyPrint',true));
end
