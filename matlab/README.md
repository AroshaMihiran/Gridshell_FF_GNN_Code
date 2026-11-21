# Gridshell form finding and sequential timber member sizing

This package contains the MATLAB form-finding solver, the fixed-geometry frame
solver and member-sizing procedure, the paper examples, and regression tests.
All paths are relative to this folder. MATLAB R2026a was used for the release
checks; form finding requires Optimization Toolbox (`fminunc`). ETABS is not
required.

## Start

Open MATLAB in this folder and run:

```matlab
setup_paths;
run_tests;
fft = run_fft('benchmark');
design = run_member_sizing('benchmark');
```

The main FFT and sizing runners write new dated folders under `outputs/`. The `.gitignore`
excludes generated results and temporary test fixtures from source control.
Original input CSVs and reference results are not overwritten.

## Form finding

```matlab
a = run_fft('benchmark');
b = run_fft('nonuniform');
c = run_fft('mixed_control');
```

The quadrilateral benchmark has 64 joints, 112 members and 49 panels. Its dead
pressure is 4.5 kPa and upward wind magnitude is 0.5 kPa. The input elastic
modulus is multiplied by 0.84 to reproduce the effective 1 GPa benchmark.

The nonuniform example has 112 joints, 280 members and 168 triangular panels.
Member areas are 25 or 64 cm², effective stiffness is 1 GPa, dead pressures are
4 or 6 kPa, and wind is 1 kPa upward. The mixed control retains the geometry
and properties but includes opposite wind signs on different panel groups.

The solver in `code/` is the same nine-file version used to generate the GNN
training data. `../release_manifest.json` records file checksums.
Low form-finding stiffness is separate from the construction stiffness used
in the later frame analysis.

Coordinates and lengths are in metres, forces in kN, elastic moduli in kN/m²,
member areas in m² and pressures in kN/m². The global z axis points upward.
Dead pressure is nonnegative and acts downward per unit current surface area.
Wind acts vertically per unit horizontal projected area:

- `WindDirection='upward'` or `'downward'`: supply a nonnegative magnitude.
- `WindDirection='signed'`: positive pressure acts downward and negative
  pressure acts upward. Signs may differ between panels.

For custom CSV inputs, call the solver directly:

```matlab
setup_paths;
options = struct('WindDirection','signed', ...
    'DeadLoadFactor',1,'WindLoadFactor',1,'IntegrationOrder',8, ...
    'MakePlots',false,'FailOnNonconvergence',false);
r = gs_fft_analysis('path_to_input_folder',options);
assert(r.success);
```

Use an included input directory as the CSV schema. Panel corner IDs are stored
as lists. An explicit boundary outline allows load-only panel closure where
physical boundary bars are absent; it does not create structural members.

Reference lengths, E and A remain fixed during FFT. Loads are frozen in each
inner energy minimization and recomputed on the updated geometry. Acceptance
requires the original relative-geometry tolerance (1e-4), refreshed free-node
force threshold (1e-6 kN + 1e-5 times the maximum free-node applied force),
successful inner minimization, and the higher-order load-integration checks.
Default FFT load factors are one. No member sizing occurs inside FFT.

`run_fft` saves the numerical state, nodal coordinates, convergence history,
timing and acceptance summary, a frame handoff, and a PNG showing the initial,
hanging and reflected geometries. The reflected picture alone is not a frame
analysis. The general `gs_options` structure retains inactive historical
sizing fields for input compatibility; they are not used by this FFT solver.

## Sequential frame sizing

```matlab
p = run_member_sizing('benchmark');
q = run_member_sizing('nonuniform');
common = run_member_sizing('benchmark','common');
independent = run_member_sizing('benchmark','independent');
omission = run_member_sizing('benchmark','omit_bending');
```

The default reproduces the paper's choices: 32 section groups for the
quadrilateral benchmark, and 280 independent physical-member sections for
the triangular roof. Grouping constrains section dimensions only; it never
symmetrizes the loads. An explicit `grouped` variant derives reflection groups
from the initial layout and supports for either example. `common` assigns one square section to all members.
`omit_bending` is a comparison experiment: it sizes without bending resistance,
then restores the full implemented checks and reports any failures. It must
not be interpreted as an accepted full-check design.

These commands use the accepted FFT handoffs in `inputs/handoffs`. To continue
from a newly computed form instead:

```matlab
fft = run_fft('benchmark');
design = run_member_sizing('benchmark','default',fft.handoffFile);
```

The hanging geometry is reflected once and becomes the stress-free frame
geometry. FFT prestress and strains are not transferred. Physical joints are
rigid frame connections; original supported translations remain restrained
and joint rotations are free degrees of freedom. Four numerical elements
per physical member resolve distributed self-weight. Member buckling lengths
remain the full physical joint-to-joint lengths.

Frame design uses C24, service class 2, Emean=11 GPa, Gmean=0.69 GPa and the
first-generation Eurocode recommended factors, without a selected National
Annex. The material data, factors, clauses and references are retained in
`design_basis/`. The five ULS combinations include permanent loading alone,
adverse/favourable permanent factors, and both signs of the prescribed wind
field. Wind reversal does not reverse dead load or member weight.

Each size update changes frame stiffness and timber self-weight, then repeats
the frame analysis and resistance evaluation at the fixed roof geometry.
There is no call back to FFT. The algorithm uses bounded member-target
bisection, damped section updates, a target utilization of 0.98, upward
rounding to 5 mm and a final reanalysis/repair stage. It seeks feasible sizes
for the implemented checks; it does not establish a global minimum or force
every utilization to equal one.

The included checks cover tension/compression, biaxial bending and assumed-K=1
member flexural buckling. Global and second-order instability, lateral
torsional buckling, shear/torsion resistance, connections, net sections,
restraint adequacy and serviceability limit compliance are not established.
These limitations are also written into the numerical exports.

Each sizing run exports member sizes/utilizations, action envelopes, frame
models and case responses, iteration histories, serviceability displacement
estimates and a final geometry PNG. Member IDs and connectivity are retained.

## Tests and independent verification

`run_tests` checks panel integration/resultants/moments, boundary identification,
signed wind, frame analytical solutions and coordinate invariance, frame
handoffs, member resistance formulas, body loads and sequential workflow rules.
It writes reports to `outputs/tests`.

`reference/expected_metrics.json` and the member CSVs hold the previously
verified benchmark values, including the bending-omission failures. They are
reference data rather than newly generated outputs.

An optional Python/NumPy checker independently reconstructs frame stiffness
from end-rotation flexibilities, integrates beam body loads, recovers actions,
and evaluates member resistance in N-mm-MPa units:

```text
python -m pip install -r validation/requirements.txt
python validation/validate_design.py outputs/sizing/<run-folder>
```

This checker supports the two packaged C24 benchmarks and their T3/Q4 panel
loads. It writes `independent_validation.json` within the selected output
folder. It does not extend the design scope listed above.

## Benchmark diagnostics and load-transfer comparison

```matlab
trace = run_benchmark_diagnostics('convergence');
comparison = run_benchmark_diagnostics('load_comparison');
```

These optional runs reproduce the inner-loop traces and the six controlled
load-rule comparisons used in the paper. Their earlier diagnostic solver is
kept separately in `examples/benchmark_diagnostics`; it is not the GNN
dataset solver. The wrapper temporarily selects that implementation and
restores the MATLAB path when finished. Run `setup_paths` before direct calls
to the main solver. The retained `GetTributaryArea.m` provides the original
benchmark load-transfer comparison; its source comments are preserved.

Diagnostic outputs use the fixed `outputs/benchmark_diagnostics` and
`outputs/load_comparison` directories and are replaced on a repeat run.
Their historical handoff format is retained for reproducibility; use
`run_fft(...).handoffFile` for the current member-sizing entry point.
