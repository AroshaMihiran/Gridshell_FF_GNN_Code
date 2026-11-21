function root=setup_paths()
%SETUP_PATHS Add this package and create local output directories.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'code'),fullfile(root,'tests'),'-begin');
assert(strcmp(which('gs_fft_analysis'),fullfile(root,'code','gs_fft_analysis.m')));
for sub={'outputs','outputs/tests'}
    folder=fullfile(root,sub{1});if ~isfolder(folder),mkdir(folder);end
end
end
