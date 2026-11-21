# Graph models

Use `predict.py` for inference, `train.py` for training/evaluation, and `tests/test_release.py` for reproduction checks. Commands and dependencies are in the repository-level README.

`model_registry.json` selects the four adopted checkpoints in `checkpoints/M1/`–`checkpoints/M4/`. `normalizer.json` is fitted on the training partition only. The six numerical modules recorded in the registry are unchanged from training. Separate metadata and original experiment fingerprints identify the dataset partitions and checkpoint configuration.

## CSV inputs

`joint.csv` contains joint IDs, initial coordinates, eight stored Laplacian modes, support flags and separate initial dead/wind nodal loads. `member.csv` contains member IDs, endpoint IDs, E, A, L0 and EA/L0. `panel.csv` contains panel IDs, cyclic JSON corner lists, initial area and dead/signed-wind pressure. The included example CSV headers give the exact field names. FFT displacement columns may be absent or blank for inference. No FFT displacement or reaction label enters the graph encoders.

Coordinates and length are in metres, E and pressure in kPa, area in m² and force in kN. Inputs must use the same Laplacian convention as the training data. The release does not regenerate positional encodings. The physical-member graph and face-to-corner incidence are distinct: face edges are load-transfer connections, not structural members.

## Reproduction checks

The examples with saved references compare CPU single-graph inference against original batched GPU outputs with absolute/relative displacement tolerances of 1e-5 m and 1e-5. Force recovery is checked at the saved coordinates with absolute/relative tolerances of 1e-8 kN and 1e-8. The tests report actual maximum discrepancies. These tolerances accommodate float32 operation ordering and do not define structural equilibrium acceptance.

`predict.py` reports the raw learned support displacements and refreshed force residuals; a successful inference run does not imply an accepted equilibrium. The outputs retain invalid geometry and support errors for assessment.

## Training data

`train.py` requires `FFT_Pilot/verification/training_index.csv`, the accepted case directories under `FFT_Pilot/cases/layout_XXXXXX/variant_YY/training/`, and their verified FFT exports and manifests. Original dataset subdirectory names are retained for compatibility. All locations are reconstructed below the supplied `--dataset` root; saved machine-local paths are ignored. The published selection and label-manifest hashes must match. New data require a separately defined training experiment and newly fitted train-only statistics.

The training driver is a portable interface to the saved training modules. `--check-data` validates the complete index and label-manifest identities and loads one graph from each partition; full graph integrity is checked as batches are read during training. It does not optimize any model. No training was performed during package preparation.
