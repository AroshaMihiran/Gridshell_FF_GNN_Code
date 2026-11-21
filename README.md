# Gridshell form finding, graph prediction and timber sizing

This repository accompanies the gridshell form-finding study. It provides the MATLAB equilibrium and frame-sizing solvers, four trained graph models, numerical examples and verification routines. Form finding and timber sizing are separate stages: sizing does not feed sections back into form finding.

## Folder guide

- `matlab/`: FFT, 3D frame analysis and sequential C24 member sizing.
- `matlab/inputs/`: complete quadrilateral benchmark, nonuniform triangular roof, mixed-wind control and accepted FFT handoffs.
- `matlab/examples/benchmark_diagnostics/`: energy traces and controlled comparisons of panel-area and corner-sharing rules.
- `matlab/tests/`: panel-load, frame, member-resistance and workflow checks.
- `matlab/validation/`: independent Python reconstruction of the sized frame in N–mm–MPa units.
- `gnn/fftgnn/`: graph construction, M1–M4 architectures, physics, training and evaluation.
- `gnn/checkpoints/M1/` through `M4/`: the four adopted `best.pt` files, exact numerical configurations, histories and test metrics.
- `gnn/examples/`: two T3/Q4 verification cases and the four illustrated test cases; each contains `joint.csv`, `member.csv` and `panel.csv`.
- `gnn/experiments/`: original experiment definitions, partition identities and label hashes.
- `validation/`: reproduction results for this package.
- `release_manifest.json`: SHA-256 checksums of distributed files.

## MATLAB examples

MATLAB R2026a and Optimization Toolbox were used. ETABS is not required. From the repository root, use MATLAB:

```matlab
cd matlab
setup_paths;
run_tests;
fft = run_fft('benchmark');
design = run_member_sizing('benchmark','default',fft.handoffFile);
```

`run_fft('nonuniform')` and `run_fft('mixed_control')` run the other supplied load fields. Each FFT export includes coordinates, member forces, refreshed-load residuals, convergence history, elapsed seconds, a frame handoff and a geometry PNG. Sizing exports sections, action envelopes, utilizations, case responses, convergence history and a sized-roof PNG. New runs use dated folders in `matlab/outputs/`.

The paper experiment entry point is:

```matlab
run_paper_experiments('form_finding');
run_paper_experiments('sizing');
run_paper_experiments('load_transfer');
run_paper_experiments('convergence');
```

The sizing study runs the default variable sections, a common section and the bending-omission comparison for both roofs. The omitted-bending sections are checked again using the full resistance equations; their reported failures are an intended experiment outcome. The default uses 32 section groups for the quadrilateral roof and one section per physical member for the triangular roof. Grouping constrains section sizes, not the applied loads.

To validate a new sized output independently, run from `matlab/`:

```text
python -m pip install -r validation/requirements.txt
python validation/validate_design.py outputs/sizing/<run-folder>
```

See `matlab/README.md` for custom CSV inputs, wind options and output definitions. Diagnostic load-transfer/energy runs use fixed diagnostic output directories; those files are replaced when the same diagnostic is repeated.

## GNN prediction

Python 3.13, PyTorch 2.11, NumPy 2.4 and pandas 2.3 were used for the package checks. CPU inference is supported; CUDA is optional. From the repository root:

```text
cd gnn
python -m pip install -r requirements.txt
python tests/test_release.py --report ../validation/inference_reproduction.json
python predict.py --model M4 --case examples/layout_000123_variant_07 --output outputs/M4_case123.csv
```

Replace M4 with M1, M2 or M3. The command uses the saved training normalizer and validation-selected checkpoint. It returns coordinates/displacements in metres and nodal force residuals in kN. The adjacent JSON gives geometry validity, free-node residuals, support displacements and elapsed seconds for model loading, graph building, inference and force recovery. This elapsed time includes setup and is not the network-only inference time reported in the paper.

The other illustrated inputs are `layout_000839_variant_06`, `layout_000027_variant_05` and `layout_000842_variant_11`. The verification examples `layout_000014_variant_04` and `layout_000017_variant_01` also include previously saved predictions, allowing direct reproduction checks.

Each model predicts all joint displacements, including supported joints. No support clamp or FFT correction is applied. Predictions must be assessed using both geometric accuracy and equilibrium/support errors before further structural use.

## Training and evaluation

The complete accepted dataset is supplied separately; the small inference examples are not a training dataset. The training loader requires the original accepted index and verified label manifests. It enforces the published geometry-separated train/validation/test partitions and reuses the original train-only normalizer.

From `gnn/`, first check the dataset without training:

```text
python -m pip install -r requirements-training.txt
python train.py --dataset /path/to/dataset --model M4 --check-data
```

Train the chosen configuration with:

```text
python train.py --dataset /path/to/dataset --model M1 --run-name M1_training
python train.py --dataset /path/to/dataset --model M2 --run-name M2_training
python train.py --dataset /path/to/dataset --model M3 --run-name M3_training
python train.py --dataset /path/to/dataset --model M4 --run-name M4_training
```

Each command trains only the named model, then evaluates its selected checkpoint. Training runs are written to `gnn/runs/`; published checkpoints are never overwritten. `--device cpu`, `--device cuda`, `--seed` and `--output` are available. To resume a new run, repeat its command with `--resume`. The released checkpoints use random seed 42. Exact rerun results can depend on the PyTorch/CUDA platform; the configuration, dataset identities and random seed are retained for reproducibility.

M1 uses width 70; M2–M4 use width 64. All use six message-passing layers, two-layer MLPs and SiLU activation. M4 has maximum physics weight 0.001, activation below 5% of initial validation data loss and a 200-epoch ramp. Its learning-rate scheduler patience is 50, whereas M1–M3 use 15. The comparison therefore includes that scheduler difference.

## Units and loading

MATLAB and GNN inputs use metres, kN, kPa (=kN/m²) and m². Global z points upward. Dead pressure is nonnegative and acts downward on current panel surface area. Signed wind pressure is positive downward and negative upward, applied vertically on horizontal projected area. The trained GNN domain contains nonnegative dead pressure and nonpositive wind pressure on initially flat T3/Q4 layouts. Wind reversal and more general polygon loading are available in MATLAB but are outside these checkpoints' training domain.

The initial low effective FFT stiffness is distinct from the C24 construction stiffness used in frame sizing. The mirrored FFT geometry becomes an initially stress-free frame; FFT prestress is not transferred. Frame joints are rigid, while the prescribed support translations are fixed and rotations remain free.

## Verification and scope

`tests/test_release.py` checks checkpoint/scaler hashes, exact state loading, partition fingerprints, eight saved T3/Q4 predictions and their force reconstruction. It also removes target displacements to verify that inference does not use labels. Numerical reproduction tolerances are documented in `gnn/README.md`.

The MATLAB tests cover panel force and moment resultants, signed wind, boundary identification, frame analytical solutions and member resistance. Member sizing implements the stated C24 compression, tension, biaxial bending and assumed-K=1 flexural-buckling checks. Global/second-order stability, lateral torsional buckling, connections, shear/torsion resistance and serviceability compliance are outside the demonstrated design scope. The design assumptions and references are in `matlab/design_basis/`.

The model weights and numerical solver cores were copied without changes. A checksum inventory records the distributed files; `python verify_files.py` checks it. Original checkpoint binaries preserve their embedded training metadata, including historical operational directory strings; inference locates files through the relative-path registry and does not use those strings. This package contains no full training dataset or hyperparameter-search checkpoints.
