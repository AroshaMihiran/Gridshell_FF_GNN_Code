function results=run_paper_experiments(study)
%RUN_PAPER_EXPERIMENTS Reproduce the numerical studies from the bundled inputs.
% study: 'form_finding', 'sizing', 'load_transfer', 'convergence', or 'all'.
if nargin<1,study='form_finding';end
root=setup_paths();study=validatestring(study,{'form_finding','sizing','load_transfer','convergence','all'});
results=struct();started=tic;
if ismember(study,{'form_finding','all'})
    for name={'benchmark','nonuniform','mixed_control'}
        r=run_fft(name{1});results.fft.(name{1})=struct('output',r.outputDirectory,'seconds',r.seconds,'accepted',r.success);
    end
end
if ismember(study,{'sizing','all'})
    for name={'benchmark','nonuniform'}
        for variant={'default','common','omit_bending'}
            r=run_member_sizing(name{1},variant{1});results.sizing.(name{1}).(variant{1})=r;
        end
    end
end
if ismember(study,{'load_transfer','all'}),results.loadTransfer=run_benchmark_diagnostics('load_comparison');end
if ismember(study,{'convergence','all'}),results.convergence=run_benchmark_diagnostics('convergence');end
results.wallSeconds=toc(started);
gs_write_json(fullfile(root,'outputs',['paper_' study '_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')) '.json']),results);
end
