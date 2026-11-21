"""Check checkpoint integrity and reproduce saved T3/Q4 test predictions."""
from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fftgnn.models import predict_coordinates
from fftgnn.physics import physics_state
from fftgnn.release import ROOT, file_sha256, input_batch, load_model, registry


REPORT = {'prediction_checks': [], 'physics_checks': []}


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.records = registry()

    def test_source_and_scaler_integrity(self):
        for name, digest in self.records['source_modules'].items():
            self.assertEqual(file_sha256(ROOT / 'fftgnn' / name), digest, name)
        self.assertEqual(file_sha256(ROOT / self.records['normalizer']), self.records['normalizer_sha256'])
        self.assertEqual(len(self.records['models']), 4)
        self.assertEqual(sum(x['primary'] for x in self.records['models']), 4)
        for experiment_name in ['baseline', 'm4_latest']:
            path = ROOT / 'experiments' / (experiment_name + '.json')
            saved = json.loads(path.read_text(encoding='utf-8'))
            import hashlib
            encoded = json.dumps(saved['definition'], sort_keys=True, default=str, allow_nan=False).encode()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), saved['fingerprint'])

    def test_all_checkpoints_and_saved_predictions(self):
        for entry in self.records['models']:
            model, normalizer, config, loaded = load_model(entry['model_id'], entry['seed'])
            self.assertEqual(sum(p.numel() for p in model.parameters()), entry['parameters'])
            self.assertEqual(config['support_mode'], 'learned')
            if entry['model_id'] == 'M4':
                self.assertTrue(entry['full_physics_weight'])
                self.assertEqual(entry['physics_weight_at_checkpoint'], .001)
                self.assertEqual(entry['scheduler_patience'], 50)
            else:
                self.assertEqual(entry['physics_weight_at_checkpoint'], 0.)
                self.assertEqual(entry['scheduler_patience'], 15)
            for sample in self.records['examples']:
                with self.subTest(model=entry['model_id'], seed=entry['seed'], sample=sample):
                    batch = input_batch(ROOT / 'examples' / sample, normalizer)
                    ref_path = ROOT / 'examples' / sample / 'reference_predictions' / f"{entry['model_id']}_seed{entry['seed']}.npz"
                    with np.load(ref_path, allow_pickle=False) as ref:
                        np.testing.assert_array_equal(batch['joint_ids'].numpy(), ref['joint_ids'])
                        np.testing.assert_array_equal(batch['x0'].numpy(), ref['initial'])
                        np.testing.assert_array_equal(batch['edge_index'].numpy(), ref['edge_index'])
                        np.testing.assert_array_equal(batch['free_mask'].numpy(), ref['free_mask'])
                        with torch.inference_mode():
                            predicted = predict_coordinates(model, batch, normalizer.length_scale) - batch['x0']
                        # Original exports used batched GPU float32 inference.
                        # CPU/single-graph GEMM ordering can differ by micrometres.
                        np.testing.assert_allclose(predicted.numpy(), ref['predicted_u'], rtol=1e-5, atol=1e-5)
                        delta = float(np.max(np.abs(predicted.numpy() - ref['predicted_u'])))
                        REPORT['prediction_checks'].append({'model':entry['model_id'],'seed':entry['seed'],'sample':sample,'maximum_difference_m':delta})
                        # Evaluate physics at exactly the saved coordinates to
                        # isolate force reconstruction from float32 inference.
                        original_coords = torch.as_tensor(ref['initial'] + ref['predicted_u'], dtype=torch.float64)
                        with torch.inference_mode():
                            physical = physics_state(original_coords, batch, order=16)
                        for actual, expected in [('residual','residual_kN'), ('external','external_pred_kN'), ('axial','axial_pred_kN')]:
                            np.testing.assert_allclose(physical[actual].numpy(), ref[expected], rtol=1e-8, atol=1e-8)
                        REPORT['physics_checks'].append({'model':entry['model_id'],'seed':entry['seed'],'sample':sample,'maximum_residual_difference_kN':float(np.max(np.abs(physical['residual'].numpy()-ref['residual_kN'])))})

    def test_unlabelled_inputs_give_the_same_features(self):
        _, normalizer, _, _ = load_model()
        source = ROOT / 'examples' / self.records['examples'][0]
        original = input_batch(source, normalizer)
        with tempfile.TemporaryDirectory(prefix='fft_gnn_inputs_') as folder:
            target = Path(folder)
            for name in ['member.csv', 'panel.csv']:
                (target/name).write_bytes((source/name).read_bytes())
            with (source/'joint.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream)
                rows=list(reader)
                fields=[f for f in reader.fieldnames if f not in ['ux_ff_m','uy_ff_m','uz_ff_m']]
            with (target/'joint.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
                writer.writeheader();writer.writerows(rows)
            unlabelled=input_batch(target, normalizer)
            for name in ['x_joint','x_homo','x_member','x_panel','x0','edge_index','incidence']:
                torch.testing.assert_close(original[name],unlabelled[name],rtol=0,atol=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseTests))
    REPORT.update(passed=outcome.wasSuccessful(),test_methods=outcome.testsRun,torch_version=torch.__version__,numpy_version=np.__version__)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(REPORT,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {len(REPORT['prediction_checks'])} predictions and {len(REPORT['physics_checks'])} force reconstructions.")
    sys.exit(0 if outcome.wasSuccessful() else 1)
