function report=run_benchmark_diagnostics(mode)
%RUN_BENCHMARK_DIAGNOSTICS Optional frozen-load traces and load-rule comparisons.
if nargin<1,mode='convergence';end
mode=validatestring(mode,{'convergence','load_comparison'});
root=setup_paths();previous=path;cleanup=onCleanup(@()path(previous));
folder=fullfile(root,'examples','benchmark_diagnostics');
addpath(folder,fullfile(folder,'code'),fullfile(folder,'legacy_loads'),'-begin');
assert(strcmp(which('gs_equilibrium'),fullfile(folder,'code','gs_equilibrium.m')));
if strcmp(mode,'convergence'),report=run_fft_diagnostics();else,report=run_load_comparison();end
end
