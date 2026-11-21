function state = gs_equilibrium(referenceNodes,bars,freeDofs,E,A,Lref,loadFunction,initialNodes,options)
%GS_EQUILIBRIUM Frozen-load energy minimization with current-geometry checks.
% Lref is NEVER reset here. Each inner solve freezes nodal loads; after it,
% loads are recomputed on the resulting geometry. Success requires a positive
% optimizer exit, small outer displacement AND a small refreshed free-DOF
% force residual. Follower-load energies are local inner objectives only.
if nargin<8 || isempty(initialNodes), initialNodes = referenceNodes; end
if nargin<9, options = struct(); end
options = completeOptions(options);
referenceState = gs_internal_state(referenceNodes,bars,E,A,Lref);
validateattributes(initialNodes,{'numeric'},{'size',size(referenceNodes),'real','finite'});
if ~isempty(freeDofs)
    validateattributes(freeDofs,{'numeric'},{'vector','integer','positive','finite'});
elseif ~isnumeric(freeDofs)
    error('GridShell:DOF','Free DOFs must be numeric indices.');
end
freeDofs = freeDofs(:);
nd = numel(referenceNodes);
if any(freeDofs>nd) || numel(unique(freeDofs))~=numel(freeDofs)
    error('GridShell:DOF','Free DOFs must be distinct indices within 1:3N.');
end
if ~isa(loadFunction,'function_handle')
    error('GridShell:Loads','loadFunction must accept current Nx3 coordinates.');
end
referenceVector = reshape(referenceNodes',[],1);
currentVector = referenceVector;
initialVector = reshape(initialNodes',[],1);
currentVector(freeDofs) = initialVector(freeDofs);
currentNodes = reshape(currentVector,3,[])';
lengthScale = max([referenceState.lengths; norm(max(referenceNodes,[],1)-min(referenceNodes,[],1))]);
history = struct('iteration',{},'relativeDisplacement',{},'residualNorm',{}, ...
    'relativeResidual',{},'forceTolerance',{},'frozenResidualNorm',{}, ...
    'loadChangeNorm',{},'exitflag',{},'innerIterations',{},'functionCount',{}, ...
    'frozenLoadPotential',{});
state = evaluatedState(currentNodes);
state.history = history;
state.exitflag = NaN;
state.relativeDisplacement = Inf;
state.iterations = 0;
state.previousNodes=currentNodes;
state.converged = false;
state.reason = 'not_started';
if isempty(freeDofs)
    state.converged = true;
    state.reason = 'all_dofs_fixed';
    state.exitflag = 1;
    state.relativeDisplacement = 0;
    return;
end
if exist('fminunc','file')~=2
    state.reason = 'fminunc_unavailable';
    return;
end
% Tight inner tolerances prevent a loose optimizer success from replacing
% the independent, dimensional equilibrium acceptance check below.
innerOptions = optimoptions('fminunc','Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true,'Display','off', ...
    'MaxIterations',options.MaxInnerIterations, ...
    'MaxFunctionEvaluations',options.MaxFunctionEvaluations, ...
    'OptimalityTolerance',min(1e-8,options.AbsoluteForceTolerance/10), ...
    'StepTolerance',1e-12,'FunctionTolerance',1e-12);
for iteration = 1:options.FFMaxIterations
    previousNodes = currentNodes;
    frozenF = state.F;
    u0 = reshape((currentNodes-referenceNodes)',[],1);
    u0 = u0(freeDofs);
    try
        [u,potential,exitflag,output] = fminunc(@frozenObjective,u0,innerOptions);
        displacement = zeros(nd,1);
        displacement(freeDofs) = u;
        currentNodes = referenceNodes+reshape(displacement,3,[])';
        candidate = evaluatedState(currentNodes);
    catch exception
        % Return the last valid state, without describing an exception as a
        % converged result or pairing an updated area with old equilibrium.
        state.converged = false;
        state.reason = ['optimizer_error: ' exception.message];
        state.errorIdentifier = exception.identifier;
        state.history = history;
        state.iterations = iteration-1;
        return;
    end
    change = sqrt(sum((currentNodes-previousNodes).^2,2));
    relativeDisplacement = max(change)/lengthScale;
    frozenResidual = candidate.internalForce-frozenF;
    record = struct('iteration',iteration,'relativeDisplacement',relativeDisplacement, ...
        'residualNorm',candidate.residualNorm,'relativeResidual',candidate.relativeResidual, ...
        'forceTolerance',candidate.forceTolerance, ...
        'frozenResidualNorm',norm(frozenResidual(freeDofs),Inf), ...
        'loadChangeNorm',norm(candidate.F-frozenF,Inf),'exitflag',exitflag, ...
        'innerIterations',output.iterations,'functionCount',output.funcCount, ...
        'frozenLoadPotential',potential);
    history(end+1) = record; %#ok<AGROW>
    state = candidate;
    state.history = history;
    state.exitflag = exitflag;
    state.optimizerOutput = output;
    state.relativeDisplacement = relativeDisplacement;
    state.previousNodes=previousNodes;
    state.iterations = iteration;
    state.converged = false;
    if ~(isfinite(exitflag) && exitflag>0)
        state.reason = 'inner_optimizer_not_converged';
        return;
    end
    if relativeDisplacement<=options.FFTolerance && state.residualNorm<=state.forceTolerance
        state.converged = true;
        state.reason = 'equilibrium_converged';
        return;
    end
    state.reason = 'equilibrium_iteration_limit';
end

    function [potential,gradient] = frozenObjective(u)
        displacement = zeros(nd,1);
        displacement(freeDofs) = u;
        nodes = referenceNodes+reshape(displacement,3,[])';
        mechanical = gs_internal_state(nodes,bars,E,A,Lref);
        potential = mechanical.strainEnergy-frozenF'*displacement;
        fullGradient = mechanical.internalForce-frozenF;
        gradient = fullGradient(freeDofs);
    end

    function s = evaluatedState(nodes)
        s = gs_internal_state(nodes,bars,E,A,Lref);
        F = loadFunction(nodes);
        if isequal(size(F),size(nodes)), F = reshape(F',[],1); else, F = F(:); end
        if numel(F)~=nd || ~isreal(F) || any(~isfinite(F))
            error('GridShell:Loads','loadFunction must return finite real loads for all 3N DOFs.');
        end
        s.F = F;
        s.residual = s.internalForce-F;
        s.reaction = s.residual;
        s.reaction(freeDofs) = 0;
        s.freeDofs = freeDofs;
        s.residualNorm = norm(s.residual(freeDofs),Inf);
        loadScale = norm(F(freeDofs),Inf);
        s.relativeResidual = s.residualNorm/max(loadScale,options.AbsoluteForceTolerance);
        s.forceTolerance = options.AbsoluteForceTolerance+options.ForceTolerance*loadScale;
        s.referenceNodes = referenceNodes;
    end
end

function options = completeOptions(options)
% Triangulation is an internally cached object, not a user scalar option.
hasTriangles = isfield(options,'PanelTriangles');
if hasTriangles, triangles=options.PanelTriangles; options=rmfield(options,'PanelTriangles'); end
options = gs_options('',options);
if hasTriangles, options.PanelTriangles=triangles; end
end
