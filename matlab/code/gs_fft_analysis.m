function result=gs_fft_analysis(inputDirectory,overrides)
%GS_FFT_ANALYSIS Axial FFT with load-only T3/Q4/polygon interpolation.
% q1 is downward force per CURRENT surface area. q2 is a NONNEGATIVE wind
% magnitude per horizontal projected area; WindDirection is explicit.
% Isolated round2: WindDirection='signed' accepts q2 positive down/negative up.
% No section sizing, reference reset, initial prestress or extra bars here.
if nargin<2,overrides=struct();end
options=gs_options(inputDirectory,overrides);
model=gs_read_model(inputDirectory,options);
options.ReferenceNodes=model.nodes;
[~,initial]=gs_panel_loads(model.nodes,model.panels,model.q1,model.q2,options);
options.PanelIntegration=initial.PanelIntegration;
Lref=sqrt(sum((model.nodes(model.bars(:,2),:)-model.nodes(model.bars(:,1),:)).^2,2));
loadFunction=@(x)gs_panel_loads(x,model.panels,model.q1,model.q2,options);
started=tic;
state=gs_equilibrium(model.nodes,model.bars,model.freeDofs, ...
    model.E*options.ElasticModulusFactor,model.A,Lref,loadFunction,model.nodes,options);
[F,loadInfo]=gs_panel_loads(state.nodes,model.panels,model.q1,model.q2,options);
residual=state.internalForce-F;
scale=norm(F(model.freeDofs),Inf);
threshold=options.AbsoluteForceTolerance+options.ForceTolerance*scale;
check=struct('passes',state.converged&&norm(residual(model.freeDofs),Inf)<=threshold, ...
    'stateConverged',state.converged,'freeForceResidual_kN',norm(residual(model.freeDofs),Inf), ...
    'forceTolerance_kN',threshold,'freshLoadsAtFinalGeometry',true,'optimizerExitFlag',state.exitflag);
% Check integration accuracy with a different, higher fixed quadrature order.
fine=options;fine.IntegrationOrder=2*options.IntegrationOrder;fine.PanelIntegration=[];
[Ffine,fineInfo]=gs_panel_loads(state.nodes,model.panels,model.q1,model.q2,fine);
quadTolerance=0.1*threshold;
quadrature=struct('order',options.IntegrationOrder,'referenceOrder',fine.IntegrationOrder, ...
    'maximumNodalDifference_kN',norm(F-Ffine,Inf),'tolerance_kN',quadTolerance, ...
    'passes',norm(F-Ffine,Inf)<=quadTolerance, ...
    'higherOrderFreeResidual_kN',norm((state.internalForce(model.freeDofs)-Ffine(model.freeDofs)),Inf), ...
    'higherOrderResidualPasses',norm((state.internalForce(model.freeDofs)-Ffine(model.freeDofs)),Inf)<=threshold, ...
    'maxPanelAreaDifference_m2',max(abs(loadInfo.panelAreas-fineInfo.panelAreas)));
result=struct('success',check.passes&&quadrature.passes&&quadrature.higherOrderResidualPasses, ...
    'reason',state.reason,'model',model,'options',options,'state',state, ...
    'loadInfo',loadInfo,'residualCheck',check,'quadratureCheck',quadrature,'seconds',toc(started));
if ~quadrature.passes||~quadrature.higherOrderResidualPasses
    result.reason='final_quadrature_accuracy_check_failed';
end
if options.FailOnNonconvergence && ~result.success
    error('Revision:FFTFailure','FFT rejected: %s. For quadrature failures increase IntegrationOrder (maximum 64) and rerun; tolerances are unchanged.',result.reason);
end
end
