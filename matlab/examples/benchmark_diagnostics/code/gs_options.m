function options = gs_options(inputDirectory, overrides)
%GS_OPTIONS Explicit settings shared by FFT and CAT. Units: m, kN, kN/m^2.
% CSV values are read first; a caller's options structure takes precedence.
if nargin < 1, inputDirectory = ''; end
if nargin < 2, overrides = struct(); end
options = struct('FFTolerance',1e-4,'ForceTolerance',1e-5, ...
    'AbsoluteForceTolerance',1e-6,'FFMaxIterations',50, ...
    'MSMaxIterations',100,'MSTolerance',0.01, ...
    'MaxRelativeAreaChange',0.01,'CoupledMaxIterations',25, ...
    'AreaDamping',0.3,'MinArea',1e-6,'MaxArea',Inf,'InitialArea',0.005, ...
    'SizingMethod','joint', ...
    'MaxInnerIterations',2000,'MaxFunctionEvaluations',100000, ...
    'ElasticModulusFactor',1,'E05Factor',0.84,'gammaM',1.3, ...
    'betaC',0.2,'EffectiveLengthFactor',1, ...
    'DeadLoadFactor',1,'WindLoadFactor',1,'WindMode','vertical', ...
    'WindDirection','downward','ReferenceNodes',[],'PanelIntegration',[], ...
    'IntegrationOrder',8, ...
    'NormalReference',[0 0 1],'GeometryRelativeTolerance',1e-10, ...
    'MakePlots',true,'PlotVisibility','on','OutputDirectory','','FailOnNonconvergence',true);
parameterFile = fullfile(inputDirectory,'analysis_parameters.csv');
if ~isempty(inputDirectory) && isfile(parameterFile)
    p = readtable(parameterFile,'Delimiter',',','TextType','string');
    if ~all(ismember({'Parameter','Value'},p.Properties.VariableNames))
        error('GridShell:Parameters','analysis_parameters.csv needs Parameter and Value.');
    end
    aliases = {'FF_Tolerance','FFTolerance'; 'FF_Max_Iter','FFMaxIterations'; ...
        'MS_Tolerance','MSTolerance'; 'MS_Max_Iter','MSMaxIterations'; ...
        'Max_Rel_Change','MaxRelativeAreaChange'; 'Coupled_Max_Iter','CoupledMaxIterations'; ...
        'Force_Tolerance','ForceTolerance'; 'Absolute_Force_Tolerance','AbsoluteForceTolerance'; ...
        'Initial_Area','InitialArea'; 'Min_Area','MinArea'; 'Max_Area','MaxArea'; ...
        'Elastic_Modulus_Factor','ElasticModulusFactor'; 'E05_Factor','E05Factor'; ...
        'Gamma_M','gammaM'; 'Beta_C','betaC'; 'Effective_Length_Factor','EffectiveLengthFactor'; ...
        'Dead_Load_Factor','DeadLoadFactor'; 'Wind_Load_Factor','WindLoadFactor'};
    names = fieldnames(options);
    for i = 1:height(p)
        name = char(p.Parameter(i));
        a = find(strcmpi(name,aliases(:,1)),1);
        if ~isempty(a), name = aliases{a,2}; end
        k = find(strcmpi(name,names),1);
        if isempty(k)
            warning('GridShell:UnusedParameter','Unused CSV parameter: %s',name);
            continue;
        end
        name = names{k};
        v = p.Value(i);
        if isnumeric(options.(name)) || islogical(options.(name))
            if ~isnumeric(v), v = str2double(string(v)); end
        else
            v = char(string(v));
        end
        options.(name) = v;
    end
end
if ~isstruct(overrides) || ~isscalar(overrides)
    error('GridShell:Options','Options must be a scalar structure.');
end
names = fieldnames(overrides);
for i = 1:numel(names)
    if ~isfield(options,names{i})
        error('GridShell:Options','Unknown option: %s',names{i});
    end
    options.(names{i}) = overrides.(names{i});
end
positive = {'FFTolerance','ForceTolerance','AbsoluteForceTolerance','MSTolerance', ...
    'MaxRelativeAreaChange','MinArea','InitialArea','ElasticModulusFactor', ...
    'E05Factor','gammaM','EffectiveLengthFactor'};
for i = 1:numel(positive)
    validateattributes(options.(positive{i}),{'numeric'},{'scalar','real','finite','positive'},mfilename,positive{i});
end
integers = {'FFMaxIterations','MSMaxIterations','CoupledMaxIterations','MaxInnerIterations','MaxFunctionEvaluations'};
for i = 1:numel(integers)
    validateattributes(options.(integers{i}),{'numeric'},{'scalar','integer','positive','finite'},mfilename,integers{i});
end
validateattributes(options.MaxArea,{'numeric'},{'scalar','real','positive','nonnan'});
validateattributes(options.AreaDamping,{'numeric'},{'scalar','real','positive','<=',1});
validateattributes(options.betaC,{'numeric'},{'scalar','real','finite','nonnegative'});
validateattributes(options.DeadLoadFactor,{'numeric'},{'scalar','real','finite','nonnegative'});
validateattributes(options.WindLoadFactor,{'numeric'},{'scalar','real','finite','nonnegative'});
if options.MaxArea < options.MinArea || options.InitialArea < options.MinArea || options.InitialArea > options.MaxArea
    error('GridShell:Options','Area bounds must include InitialArea.');
end
options.WindMode = validatestring(options.WindMode,{'vertical'});
options.WindDirection = validatestring(options.WindDirection,{'downward','upward'});
validateattributes(options.IntegrationOrder,{'numeric'},{'scalar','integer','>=',2,'<=',64});
if ~isempty(options.ReferenceNodes)
    validateattributes(options.ReferenceNodes,{'numeric'},{'2d','ncols',3,'real','finite'});
end
options.SizingMethod = validatestring(options.SizingMethod,{'joint','ratio'});
options.PlotVisibility = validatestring(options.PlotVisibility,{'on','off'});
validateattributes(options.NormalReference,{'numeric'},{'2d','ncols',3,'real','finite'});
validateattributes(options.GeometryRelativeTolerance,{'numeric'},{'scalar','real','positive','<',0.01});
validateattributes(options.MakePlots,{'logical','numeric'},{'scalar','binary'});
validateattributes(options.FailOnNonconvergence,{'logical','numeric'},{'scalar','binary'});
if ~(ischar(options.OutputDirectory) || (isstring(options.OutputDirectory) && isscalar(options.OutputDirectory)))
    error('GridShell:Options','OutputDirectory must be text.');
end
end
