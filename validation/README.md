# Package reproduction checks

The checks in this folder were run on 20 November 2026 without retraining.

- `matlab_tests.json`: all eight MATLAB test groups passed.
- `matlab_reproduction.json`: fresh benchmark FFT and sequential member sizing, with unchanged convergence criteria.
- `independent_frame_verification.json`: 37 independent frame/load comparisons and C24 member-resistance audit passed.
- `inference_reproduction.json`: eight CPU predictions (M1–M4 on T3/Q4 examples) and eight force reconstructions matched saved references.
- `training_data_preflight.json`: all 10,075 accepted sample/split identities and label-manifest hashes matched; one verified graph per partition loaded. No training was started.
- `reproduction_summary.json`: maximum reproduction differences. FFT height, sizing volume and maximum utilization matched the original packaged reference values exactly on this installation.

The model outputs remain raw surrogate predictions. Reproducing a prediction does not imply that it satisfies equilibrium or support constraints. Timings in these records describe this local check, not a new performance benchmark.
