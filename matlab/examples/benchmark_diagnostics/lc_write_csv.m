function lc_write_csv(path,headers,values)
% Explicit 17-significant-digit export, including IDs and logical values.
assert(size(values,2)==numel(headers));
fid=fopen(path,'w');assert(fid>0);cleaner=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',strjoin(headers,','));
fmt=[repmat('%.17g,',1,size(values,2)-1) '%.17g\n'];
fprintf(fid,fmt,values');
end
