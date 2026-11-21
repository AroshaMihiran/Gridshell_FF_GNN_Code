"""Portable loading and CSV inference for the released FFT surrogate models."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .csv_io import load_graph
from .data import Normalizer
from .models import FormFindingGNN, predict_coordinates
from .physics import physics_state


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry(root=ROOT):
    return json.loads((Path(root) / 'model_registry.json').read_text(encoding='utf-8'))


def load_model(model_id='M4', seed=42, root=ROOT, device='cpu'):
    """Load a validation-selected checkpoint and its training-only normalizer.

    Historical directory strings embedded in the original checkpoint are not
    used. The release registry supplies portable paths; model configuration is
    checked against the original configuration except those directory strings.
    """
    root = Path(root).resolve()
    records = registry(root)
    matches = [r for r in records['models'] if r['model_id'] == model_id and r['seed'] == seed]
    if len(matches) != 1:
        raise ValueError(f'No unique released model for {model_id}, seed {seed}')
    entry = matches[0]
    checkpoint_path = root / entry['checkpoint']
    if file_sha256(checkpoint_path) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs from the release registry.')
    normalizer_path = root / records['normalizer']
    if file_sha256(normalizer_path) != records['normalizer_sha256']:
        raise ValueError('Normalizer checksum differs from the release registry.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = json.loads((root / entry['config']).read_text(encoding='utf-8'))
    original = {k: v for k, v in checkpoint['config'].items() if k not in ('dataset_root', 'output_root')}
    if config != original or checkpoint['fingerprint'] != entry['experiment_fingerprint']:
        raise ValueError('Checkpoint configuration or experiment identity has changed.')
    if checkpoint['model_id'] != model_id or checkpoint['seed'] != seed:
        raise ValueError('Checkpoint model/seed identity does not match the registry.')
    model = FormFindingGNN(model_id, config)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    normalizer = Normalizer.from_dict(json.loads(normalizer_path.read_text(encoding='utf-8')))
    return model, normalizer, config, entry


def input_batch(case_directory, normalizer, device='cpu'):
    """Build one graph without using FFT targets or reaction labels as inputs.

    Accepts the original joint/member/panel CSV schema. FFT displacement columns
    may be fully absent or blank. The eight stored Laplacian modes remain part
    of the schema; no normalization or positional encoding is fitted here.
    """
    graph = load_graph(case_directory, require_targets=False)
    x0, members, edges = graph['initial_coords'], graph['member_features'], graph['physical_edge_index']
    if np.any(x0[:, 2] != 0):
        raise ValueError('The released models require an initially flat layout with z0=0.')
    if len(np.unique(x0, axis=0)) != len(x0):
        raise ValueError('Duplicate initial joint coordinates.')
    if not graph['free_mask'].any():
        raise ValueError('The input must contain at least one free joint.')
    measured = np.linalg.norm(x0[edges[1]] - x0[edges[0]], axis=1)
    if not np.allclose(measured, members[:, 2], rtol=1e-10, atol=1e-10):
        raise ValueError('Member L0 does not match the initial geometry.')
    if not np.allclose(members[:, 0] * members[:, 1] / members[:, 2], members[:, 3], rtol=1e-10, atol=1e-10):
        raise ValueError('Member EA/L0 is inconsistent.')
    corners = graph['panel_corners']
    if any(len(p) not in (3, 4) for p in corners):
        raise NotImplementedError('The released models use T3/Q4 panels only.')
    if len(set(tuple(sorted(map(int, p))) for p in corners)) != len(corners):
        raise ValueError('Duplicate load panels.')
    double = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.float64)
    integer = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.long)
    lpe_dim = normalizer.lpe_dim
    cols = [0, 1] + list(range(2, 2 + lpe_dim)) + [10]
    batch = {
        'x0': double(x0), 'node_raw': double(graph['node_features'][:, cols]),
        'member_raw': double(members), 'panel_raw': double(graph['panel_features']),
        'PD0': double(graph['PD0_kN']), 'PW0': double(graph['PW0_kN']),
        'free_mask': torch.as_tensor(graph['free_mask'], dtype=torch.bool),
        'edge_index': integer(edges), 'incidence': integer(graph['panel_incidence_index']),
        'joint_ids': integer(graph['joint_ids']), 'member_ids': integer(graph['member_ids']),
        'node_batch': torch.zeros(len(x0), dtype=torch.long), 'num_graphs': 1,
        'lpe_dim': lpe_dim,
    }
    for size, name in [(3, 'tri'), (4, 'quad')]:
        ids = [i for i, c in enumerate(corners) if len(c) == size]
        batch[name + '_panel'] = integer(ids)
        batch[name + '_index'] = integer(np.asarray([corners[i] for i in ids], dtype=np.int64).reshape(-1, size))
    batch.update(normalizer.transform(batch))
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict_case(case_directory, model_id='M4', seed=42, root=ROOT, device='cpu'):
    model, normalizer, config, entry = load_model(model_id, seed, root, device)
    batch = input_batch(case_directory, normalizer, device)
    with torch.inference_mode():
        coords = predict_coordinates(model, batch, normalizer.length_scale)
        forces = physics_state(coords, batch, order=config['evaluation_integration_order'])
    return coords, forces, batch, entry
