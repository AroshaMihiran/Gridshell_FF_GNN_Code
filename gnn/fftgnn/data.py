"""Accepted FFT labels, leakage-safe partitions and train-only normalization.

No routine in this module writes to the source dataset. Physical tensors are
float64; encoded input features are float32. Member connectivity is unique and
undirected. Panel incidence is directed face-to-joint, never a physical bar.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .csv_io import load_graph
from .reactions import read_verified_reactions

SPLITS = ("train", "validation", "test")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_partitions(frame, require_all=True):
    if frame.empty or frame.sample_id.duplicated().any():
        raise ValueError("The selected index is empty or has duplicate sample IDs.")
    if not set(frame.split).issubset(SPLITS):
        raise ValueError("Split names must be train, validation or test.")
    for name in ("base_layout_id", "normalized_geometry_group"):
        if frame.groupby(name).split.nunique().max() > 1:
            raise ValueError(f"Data leakage: {name} crosses partitions.")
    if require_all and set(frame.split) != set(SPLITS):
        raise ValueError("Selection must contain train, validation and test; increase max_layouts.")


def load_index(dataset_root):
    """Read ONLY the independently accepted training index.

    Published absolute paths are not trusted as machine-local paths. Locations
    are reconstructed from validated IDs below ``dataset_root/FFT_Pilot/cases``;
    this makes a copied dataset portable and prevents path traversal.
    """
    root = Path(dataset_root).expanduser().resolve()
    path = root / "FFT_Pilot/verification/training_index.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"sample_id", "base_layout_id", "variant_id", "split", "normalized_geometry_group", "independently_verified"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Accepted index is missing: {sorted(required - set(frame.columns))}")
    if not frame.independently_verified.astype(str).str.lower().eq("true").all():
        raise ValueError("Index contains a sample without independent verification.")
    if not frame.base_layout_id.str.fullmatch(r"layout_\d{6}").all():
        raise ValueError("Invalid layout ID in the accepted index.")
    variant = pd.to_numeric(frame.variant_id, errors="raise")
    if ((variant != variant.astype(int)) | (variant < 1)).any():
        raise ValueError("Variant IDs must be positive integers.")
    frame["variant_id"] = variant.astype(int)
    expected = frame.base_layout_id + "_variant_" + frame.variant_id.map(lambda x: f"{x:02d}")
    if not expected.eq(frame.sample_id).all():
        raise ValueError("Sample IDs disagree with layout/variant IDs.")
    if not frame.normalized_geometry_group.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid normalized geometry group.")
    cases_root = (root / "FFT_Pilot/cases").resolve()
    case_paths = [(cases_root / r.base_layout_id / f"variant_{r.variant_id:02d}").resolve() for r in frame.itertuples()]
    if any(not p.is_relative_to(cases_root) for p in case_paths):
        raise ValueError("A case path escapes the dataset.")
    frame["case_directory"] = [str(p) for p in case_paths]
    frame["training_directory"] = [str(p / "training") for p in case_paths]
    frame["label_manifest"] = [str(p / "training/label_manifest.json") for p in case_paths]
    frame["source_split"] = frame.split
    frame["dataset_root"] = str(root)
    layout_file = root / "dataset/layouts.csv"
    family = {}
    if layout_file.exists():
        layouts = pd.read_csv(layout_file)
        key = "layout_id" if "layout_id" in layouts else "base_layout_id"
        if key in layouts and "family" in layouts:
            family = dict(zip(layouts[key], layouts.family))
    frame["family"] = frame.base_layout_id.map(family).fillna("unspecified")
    _assert_partitions(frame)
    frame.attrs["index_sha256"] = _sha(path)
    return frame.sort_values(["base_layout_id", "variant_id"]).reset_index(drop=True)


def _balanced_layout_subset(frame, count, rng):
    """Round-robin random layout selection across split/family strata.

    All accepted variants of selected layouts are retained unless separately
    capped. Geometry-group identities keep their already assigned partition.
    """
    layouts = frame[["base_layout_id", "split", "family"]].drop_duplicates()
    if count >= len(layouts):
        return set(layouts.base_layout_id)
    if count < 3:
        raise ValueError("max_layouts must be at least 3 for three nonempty splits.")
    selected = []
    # At least one layout per split, then proportional allocation by split.
    allocations = {s: 1 for s in SPLITS}
    populations = {s: int((layouts.split == s).sum()) for s in SPLITS}
    for _ in range(count - 3):
        candidates = [s for s in SPLITS if allocations[s] < populations[s]]
        s = max(candidates, key=lambda x: populations[x] / (allocations[x] + .5))
        allocations[s] += 1
    for s in SPLITS:
        strata = []
        for _, rows in layouts[layouts.split == s].groupby("family", sort=True):
            values = rows.base_layout_id.to_numpy().copy()
            rng.shuffle(values)
            strata.append(list(values))
        rng.shuffle(strata)
        needed = allocations[s]
        while needed:
            for values in strata:
                if values and needed:
                    selected.append(values.pop())
                    needed -= 1
    return set(selected)


def split_index(index, mode="existing", fractions=(.8, .1, .1), seed=42,
                max_layouts=None, max_variants_per_layout=None):
    """Return selected rows with an immutable layout/group-level split.

    ``existing`` preserves the published 80/10/10 layout assignment. In
    ``random_group`` mode, ratios apply approximately to geometry-group counts
    (not individual cases). Existing split identities remain in source_split
    for checking the original label manifest.
    """
    frame = index.copy()
    if "source_split" not in frame:
        frame["source_split"] = frame.split
    if "family" not in frame:
        frame["family"] = "unspecified"
    _assert_partitions(frame)
    rng = np.random.default_rng(seed)
    if mode not in ("existing", "random_group"):
        raise ValueError("split mode must be existing or random_group.")
    fractions = np.asarray(fractions, float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions must sum to one.")
    if mode == "random_group":
        groups = np.array(sorted(frame.normalized_geometry_group.unique()))
        if len(groups) < 3:
            raise ValueError("At least three independent geometry groups are required.")
        rng.shuffle(groups)
        counts = np.maximum(1, np.floor(len(groups) * fractions).astype(int))
        while counts.sum() > len(groups):
            candidates = np.flatnonzero(counts > 1)
            counts[candidates[np.argmax(counts[candidates])]] -= 1
        while counts.sum() < len(groups):
            counts[np.argmax(len(groups) * fractions - counts)] += 1
        mapping, start = {}, 0
        for s, count in zip(SPLITS, counts):
            mapping.update({g: s for g in groups[start:start + count]})
            start += count
        frame["split"] = frame.normalized_geometry_group.map(mapping)
    if max_layouts is not None:
        if isinstance(max_layouts, bool) or int(max_layouts) != max_layouts or max_layouts < 3:
            raise ValueError("max_layouts must be None or an integer >= 3.")
        chosen = _balanced_layout_subset(frame, int(max_layouts), rng)
        frame = frame[frame.base_layout_id.isin(chosen)].copy()
    if max_variants_per_layout is not None:
        if isinstance(max_variants_per_layout, bool) or int(max_variants_per_layout) != max_variants_per_layout or max_variants_per_layout < 1:
            raise ValueError("max_variants_per_layout must be None or a positive integer.")
        pieces = []
        for _, rows in frame.groupby("base_layout_id", sort=True):
            n = min(len(rows), int(max_variants_per_layout))
            pieces.append(rows.iloc[np.sort(rng.choice(len(rows), n, replace=False))])
        frame = pd.concat(pieces)
    frame = frame.sort_values(["base_layout_id", "variant_id"]).reset_index(drop=True)
    _assert_partitions(frame)
    frame.attrs.update(index.attrs)
    return frame


def split_fingerprint(index):
    columns = ["sample_id", "split", "normalized_geometry_group"]
    payload = index[columns].sort_values("sample_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def feature_names(lpe_dim=8):
    joint = ["x0_m", "y0_m"] + [f"lap_pe_{i:02d}" for i in range(1, lpe_dim + 1)] + ["support"]
    return {"joint": joint, "homo": joint + ["PD0_kN", "PW0_kN"],
            "member": ["E_kPa", "A_m2", "L0_m", "EA_over_L0_kN_per_m"],
            "panel": ["qD_kPa", "qW_kPa", "initial_area_m2"]}


class GraphDataset(Dataset):
    """Load accepted CSV labels lazily, verifying their hashes once per cache load.

    Individual physical tensors remain double precision. Do not mutate returned
    tensors in-place: cached raw samples are shared. collate_graphs makes fresh
    packed tensors, which are safe to augment.

    Support-reaction targets are always loaded from the verified FFT export.
    They are labels only, never encoder features or normalization inputs.
    """
    def __init__(self, index, normalizer=None, lpe_dim=8, cache_size=256, verify_hashes=True):
        if isinstance(lpe_dim, bool) or not 0 <= lpe_dim <= 8 or int(lpe_dim) != lpe_dim:
            raise ValueError("lpe_dim must be an integer from 0 through the 8 stored modes.")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative.")
        self.index = index.reset_index(drop=True).copy()
        self.normalizer, self.lpe_dim = normalizer, int(lpe_dim)
        self.cache_size, self.verify_hashes = int(cache_size), bool(verify_hashes)
        self._cache = OrderedDict()
        if normalizer is not None and normalizer.lpe_dim != self.lpe_dim:
            raise ValueError("Normalizer and dataset LPE dimensions disagree.")

    def __len__(self):
        return len(self.index)

    def _read(self, i):
        row = self.index.iloc[i]
        directory = Path(row.training_directory)
        manifest_path = directory / "label_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "independently_verified_fft_labels":
            raise ValueError(f"Unverified labels: {directory}")
        for key in ("sample_id", "base_layout_id", "normalized_geometry_group"):
            if manifest.get(key) != row[key]:
                raise ValueError(f"Manifest/index mismatch: {key} for {row.sample_id}")
        if manifest.get("split") != row.get("source_split", row.split):
            raise ValueError("Original label split differs from its accepted index.")
        if int(manifest.get("variant_id", -1)) != int(row.variant_id):
            raise ValueError("Manifest variant identity mismatch.")
        if self.verify_hashes:
            for name in ("joint.csv", "member.csv", "panel.csv"):
                if _sha(directory / name) != manifest.get("training_csv_sha256", {}).get(name):
                    raise ValueError(f"Published-label hash mismatch: {directory / name}")
        graph = load_graph(directory, require_targets=True)
        x0 = graph["initial_coords"]
        edges = graph["physical_edge_index"]
        members = graph["member_features"]
        measured = np.linalg.norm(x0[edges[1]] - x0[edges[0]], axis=1)
        if not np.allclose(measured, members[:, 2], rtol=1e-10, atol=1e-10):
            raise ValueError("Member L0 does not match the original coordinates.")
        if not np.allclose(members[:, 0] * members[:, 1] / members[:, 2], members[:, 3], rtol=1e-10, atol=1e-10):
            raise ValueError("Member EA/L0 input is inconsistent.")
        if not graph["free_mask"].any():
            raise ValueError("A training graph must contain at least one free joint.")
        if np.any(x0[:, 2] != 0):
            raise ValueError("The present model inputs require initial z=0.")
        if len(np.unique(x0, axis=0)) != len(x0):
            raise ValueError("Duplicate original joint coordinates.")
        corners = graph["panel_corners"]
        if any(len(p) not in (3, 4) for p in corners):
            raise NotImplementedError("This dataset physics implementation supports T3/Q4 panels; no polygon is silently triangulated.")
        keys = [tuple(sorted(map(int, p))) for p in corners]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate load panels.")
        as_double = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.float64)
        as_long = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.long)
        selected_cols = [0, 1] + list(range(2, 2 + self.lpe_dim)) + [10]
        sample = {
            "sample_id": str(row.sample_id), "base_layout_id": str(row.base_layout_id),
            "normalized_geometry_group": str(row.normalized_geometry_group), "split": str(row.split),
            "family": str(row.get("family", "unspecified")), "case_directory": str(row.case_directory),
            "training_directory": str(directory), "label_manifest": manifest,
            "x0": as_double(x0), "target_u": as_double(graph["target_displacements"]),
            "free_mask": torch.as_tensor(graph["free_mask"], dtype=torch.bool),
            "edge_index": as_long(edges), "member_raw": as_double(members),
            "panel_raw": as_double(graph["panel_features"]),
            "node_raw": as_double(graph["node_features"][:, selected_cols]),
            "joint_ids": as_long(graph["joint_ids"]), "member_ids": as_long(graph["member_ids"]),
            "panel_ids": as_long(graph["panel_ids"]), "panel_corners": [as_long(p) for p in corners],
            "incidence": as_long(graph["panel_incidence_index"]),
            "PD0": as_double(graph["PD0_kN"]), "PW0": as_double(graph["PW0_kN"]),
            "lpe_dim": self.lpe_dim,
        }
        sample["reaction_mask"] = ~sample["free_mask"]
        sample["reaction_target"] = read_verified_reactions(
            row.case_directory, sample["joint_ids"], sample["free_mask"])
        for size, name in ((3, "tri"), (4, "quad")):
            panel_indices = [j for j, p in enumerate(corners) if len(p) == size]
            sample[name + "_panel"] = as_long(panel_indices)
            sample[name + "_index"] = as_long(np.asarray([corners[j] for j in panel_indices], dtype=np.int64).reshape(-1, size))
        return sample

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i in self._cache:
            sample = self._cache.pop(i)
            self._cache[i] = sample
        else:
            sample = self._read(i)
            if self.cache_size:
                self._cache[i] = sample
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        sample = sample.copy()
        sample["normalization_applied"] = self.normalizer is not None
        if self.normalizer is not None:
            sample.update(self.normalizer.transform(sample))
        else:
            sample.update(x_joint=sample["node_raw"].float(),
                          x_homo=torch.cat((sample["node_raw"], sample["PD0"][:, None], sample["PW0"][:, None]), 1).float(),
                          x_member=sample["member_raw"].float(), x_panel=sample["panel_raw"].float())
        return sample


class _Moments:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def update(self, x):
        x = x.detach().cpu().numpy().astype(np.float64, copy=False)
        n = len(x)
        if not n:
            return
        mean = x.mean(axis=0)
        m2 = np.square(x - mean).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mean, m2
        else:
            delta = mean - self.mean
            total = self.n + n
            self.m2 += m2 + delta * delta * self.n * n / total
            self.mean += delta * n / total
            self.n = total

    def finish(self):
        std = np.sqrt(np.maximum(self.m2 / self.n, 0))
        std[std < 1e-12] = 1.
        return self.mean.tolist(), std.tolist()


class Normalizer:
    """Feature standardization fitted strictly on the training partition.

    Feature moments weight each entity equally within its node/edge/face type.
    F_tr weights structures equally, then joints within each structure. L_tr is
    the RMS reference member length over training members. Both scales use
    inputs only; output means/variances are never fitted to held-out targets.
    """
    def __init__(self, stats, length_scale, force_scale, lpe_dim=8, training_ids=None):
        self.stats = stats
        self.length_scale, self.force_scale = float(length_scale), float(force_scale)
        self.lpe_dim = int(lpe_dim)
        self.training_ids = list(training_ids or [])
        if not (np.isfinite(self.length_scale) and self.length_scale > 0 and np.isfinite(self.force_scale) and self.force_scale > 0):
            raise ValueError("Training length and force scales must be finite and positive.")

    @classmethod
    def fit(cls, dataset):
        if not len(dataset) or set(dataset.index.split) != {"train"}:
            raise ValueError("Normalizer.fit requires a nonempty training-only dataset.")
        moments = {name: _Moments() for name in ("joint", "loads", "member", "panel")}
        length_sum, member_count, force_sum, ids = 0., 0, 0., []
        for sample in dataset:
            for name, values in (("joint", sample["node_raw"]), ("loads", torch.stack((sample["PD0"], sample["PW0"]), 1)),
                                 ("member", sample["member_raw"]), ("panel", sample["panel_raw"])):
                moments[name].update(values)
            lengths = sample["member_raw"][:, 2]
            length_sum += float(lengths.square().sum())
            member_count += len(lengths)
            force_sum += float((sample["PD0"].square() + sample["PW0"].square()).mean())
            ids.append(sample["sample_id"])
        stats = {k: dict(zip(("mean", "std"), v.finish())) for k, v in moments.items()}
        stats["joint"]["mean"][-1], stats["joint"]["std"][-1] = 0., 1.
        return cls(stats, np.sqrt(length_sum / member_count), np.sqrt(force_sum / len(ids)), dataset.lpe_dim, ids)

    def transform(self, sample):
        def standardize(x, kind):
            s = self.stats[kind]
            mean = torch.tensor(s["mean"], dtype=x.dtype, device=x.device)
            std = torch.tensor(s["std"], dtype=x.dtype, device=x.device)
            return ((x - mean) / std).float()
        node = standardize(sample["node_raw"], "joint")
        load = standardize(torch.stack((sample["PD0"], sample["PW0"]), 1), "loads")
        return {"x_joint": node, "x_homo": torch.cat((node, load), 1),
                "x_member": standardize(sample["member_raw"], "member"),
                "x_panel": standardize(sample["panel_raw"], "panel")}

    def to_dict(self):
        return {"schema_version": 1, "stats": self.stats, "length_scale": self.length_scale,
                "force_scale": self.force_scale, "lpe_dim": self.lpe_dim,
                "training_ids": self.training_ids,
                "normalization_policy": "train-only entity feature moments; support unchanged; no coordinate centering per graph"}

    def feature_stats(self, name):
        """Return mean/std in encoded-column order, useful for raw LPE sign flips."""
        mapping = {"x_joint": "joint", "x_member": "member", "x_panel": "panel"}
        if name == "x_homo":
            mean = self.stats["joint"]["mean"] + self.stats["loads"]["mean"]
            std = self.stats["joint"]["std"] + self.stats["loads"]["std"]
        elif name in mapping:
            mean, std = self.stats[mapping[name]]["mean"], self.stats[mapping[name]]["std"]
        else:
            raise KeyError(name)
        return np.asarray(mean, float), np.asarray(std, float)

    @classmethod
    def from_dict(cls, obj):
        if obj.get("schema_version") != 1:
            raise ValueError("Unknown normalization schema.")
        return cls(obj["stats"], obj["length_scale"], obj["force_scale"], obj["lpe_dim"], obj["training_ids"])


def collate_graphs(samples):
    """Pack disjoint graphs; all relation indices receive the proper offsets."""
    if not samples:
        raise ValueError("Cannot pack an empty batch.")
    if len({s["lpe_dim"] for s in samples}) != 1:
        raise ValueError("Mixed LPE dimensions in one batch.")
    names = ("x0", "target_u", "free_mask", "reaction_mask", "reaction_target", "member_raw", "panel_raw", "node_raw", "PD0", "PW0",
             "joint_ids", "member_ids", "panel_ids", "x_joint", "x_homo", "x_member", "x_panel")
    output = {name: torch.cat([s[name] for s in samples], dim=0) for name in names}
    edges, incidence, tri, quad, tri_p, quad_p = [], [], [], [], [], []
    node_batch, member_batch, panel_batch, ptr, panel_ptr, member_ptr = [], [], [], [0], [0], [0]
    for g, sample in enumerate(samples):
        n, m, p = len(sample["x0"]), len(sample["member_raw"]), len(sample["panel_raw"])
        edges.append(sample["edge_index"] + ptr[-1])
        incidence.append(sample["incidence"] + torch.tensor([[panel_ptr[-1]], [ptr[-1]]]))
        tri.append(sample["tri_index"] + ptr[-1]); quad.append(sample["quad_index"] + ptr[-1])
        tri_p.append(sample["tri_panel"] + panel_ptr[-1]); quad_p.append(sample["quad_panel"] + panel_ptr[-1])
        node_batch.append(torch.full((n,), g, dtype=torch.long))
        member_batch.append(torch.full((m,), g, dtype=torch.long))
        panel_batch.append(torch.full((p,), g, dtype=torch.long))
        ptr.append(ptr[-1] + n); member_ptr.append(member_ptr[-1] + m); panel_ptr.append(panel_ptr[-1] + p)
    output.update(edge_index=torch.cat(edges, 1), incidence=torch.cat(incidence, 1),
                  tri_index=torch.cat(tri), quad_index=torch.cat(quad),
                  tri_panel=torch.cat(tri_p), quad_panel=torch.cat(quad_p),
                  node_batch=torch.cat(node_batch), member_batch=torch.cat(member_batch), panel_batch=torch.cat(panel_batch),
                  ptr=torch.tensor(ptr), member_ptr=torch.tensor(member_ptr), panel_ptr=torch.tensor(panel_ptr),
                  num_graphs=len(samples), lpe_dim=samples[0]["lpe_dim"], samples=samples,
                  sample_ids=[s["sample_id"] for s in samples])
    return output


def move_batch(batch, device):
    """Transfer packed tensors only; GUI/source metadata stays on the CPU."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
