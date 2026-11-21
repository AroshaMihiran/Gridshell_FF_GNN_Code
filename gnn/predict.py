"""Predict one CSV graph using a released model; no training or FFT correction."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fftgnn.release import ROOT, predict_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['M1', 'M2', 'M3', 'M4'], default='M4')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--case', type=Path, default=ROOT / 'examples/layout_000014_variant_04')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/prediction.csv')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    coords, forces, batch, entry = predict_case(args.case, args.model, args.seed, device=args.device)
    elapsed = time.perf_counter() - started
    xyz = coords.cpu().numpy()
    u = xyz - batch['x0'].cpu().numpy()
    free = batch['free_mask'].cpu().numpy()
    residual = forces['residual'].cpu().numpy()
    output = pd.DataFrame({'joint_id': batch['joint_ids'].cpu().numpy(), 'support': (~free).astype(int)})
    for j, axis in enumerate('xyz'):
        output[axis + '_pred_m'] = xyz[:, j]
        output['u' + axis + '_pred_m'] = u[:, j]
        output['residual_' + axis + '_kN'] = residual[:, j]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    diagnostics = {
        'model_id': args.model, 'seed': args.seed, 'checkpoint_epoch': entry['checkpoint_epoch'],
        'prediction_pipeline_seconds': elapsed,
        'timing_scope': 'Checkpoint loading, CSV graph construction, network inference and force reconstruction; excludes CSV writing.',
        'joint_count': len(xyz), 'valid_geometry': bool(forces['valid_graph'].all()),
        'free_residual_rms_kN': float(np.sqrt(np.mean(residual[free] ** 2))),
        'free_residual_max_kN': float(np.max(np.abs(residual[free]))),
        'maximum_support_displacement_m': float(np.linalg.norm(u[~free], axis=1).max()) if (~free).any() else None,
        'support_mode': entry['support_mode'],
        'note': 'Raw surrogate prediction. Support displacements are learned and are not clamped; no FFT equilibrium correction is applied.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(diagnostics, indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
