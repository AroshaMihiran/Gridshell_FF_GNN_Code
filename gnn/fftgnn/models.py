"""Exact manuscript gather/scatter architectures, without a PyG dependency.

Each undirected physical member is stored once. Panel conditioning occurs
once before structural layers. Global means are separate for every graph.
"""
import torch
from torch import nn


MODEL_NAMES = {"M1": "Homogeneous", "M2": "Heterogeneous", "M3": "Heterogeneous + global", "M4": "Heterogeneous + global + physics"}


def mlp(n_in, n_out, hidden, depth, activation="silu", dropout=0.):
    act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    modules = []
    for i in range(depth):
        modules.append(nn.Linear(n_in if i == 0 else hidden, n_out if i == depth - 1 else hidden))
        if i < depth - 1:
            modules.append(act())
            if dropout:
                modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


def segment_sum(values, index, size):
    return values.new_zeros((size, *values.shape[1:])).index_add(0, index, values)


class FormFindingGNN(nn.Module):
    def __init__(self, model_id, cfg):
        super().__init__()
        if model_id not in MODEL_NAMES:
            raise ValueError(model_id)
        self.model_id, self.lpe_dim = model_id, cfg["lpe_dim"]
        self.support_mode = cfg["support_mode"]
        if self.support_mode not in ("learned", "hard"):
            raise ValueError("support_mode must be learned or hard")
        self.heterogeneous, self.global_pooling = model_id != "M1", model_id in ("M3", "M4")
        h = cfg.get("hidden_dims_by_model", {}).get(model_id, cfg["hidden_dim"])
        self.hidden_dim = h
        depth = cfg["mlp_depth"]
        def make(a, b=h):
            return mlp(a, b, h, depth, cfg["activation"], cfg["dropout"])
        self.joint_encoder = make(3 + self.lpe_dim + (0 if self.heterogeneous else 2))
        self.member_encoder = make(4)
        if self.heterogeneous:
            self.panel_encoder = make(3)
            self.face_joint = make(2 * h)
            self.condition = make(2 * h)
        self.gathers = nn.ModuleList(make(2 * h) for _ in range(cfg["layers"]))
        self.updates = nn.ModuleList(make(2 * h) for _ in range(cfg["layers"]))
        self.decoder = make(h, 3)

    def forward(self, batch):
        x = batch["x_joint"] if self.heterogeneous else batch["x_homo"]
        h = self.joint_encoder(x.float())
        ze = self.member_encoder(batch["x_member"].float())
        if self.heterogeneous:
            zf = self.panel_encoder(batch["x_panel"].float())
            face, joint = batch["incidence"]
            messages = self.face_joint(torch.cat((h[joint], zf[face]), dim=-1))
            h = self.condition(torch.cat((h, segment_sum(messages, joint, len(h))), dim=-1))
        u, v = batch["edge_index"]
        for gather, update in zip(self.gathers, self.updates):
            message = gather(torch.cat(((h[u] + h[v]) * .5, ze), dim=-1))
            aggregate = segment_sum(message, u, len(h)) + segment_sum(message, v, len(h))
            h = update(torch.cat((h, aggregate), dim=-1))
            if self.global_pooling:
                count = torch.bincount(batch["node_batch"], minlength=batch["num_graphs"]).clamp_min(1)
                pooled = segment_sum(h, batch["node_batch"], batch["num_graphs"]) / count[:, None]
                h = h + pooled[batch["node_batch"]]
        displacement = self.decoder(h)
        if self.support_mode == "hard":
            displacement = displacement * batch["free_mask"][:, None]
        return displacement


def predict_coordinates(model, batch, length_scale):
    normalized_u = model(batch)
    return batch["x0"] + normalized_u.to(torch.float64) * length_scale


def augment_lpe(batch, lpe_dim, normalizer):
    """Flip raw eigenvector signs per graph/mode, then apply the saved scaler.

    This must not merely negate a standardized feature whose fitted mean may
    be nonzero. Repeated-eigenspace rotations are not claimed to be solved.
    """
    signs = torch.randint(0, 2, (batch["num_graphs"], lpe_dim), device=batch["x_joint"].device) * 2 - 1
    out = dict(batch)
    for key in ("x_joint", "x_homo"):
        value = batch[key].clone()
        # Scaler vectors are supplied through a public accessor, not refitted.
        mean, scale = normalizer.feature_stats(key)
        mean = torch.as_tensor(mean[2:2+lpe_dim], device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(scale[2:2+lpe_dim], device=value.device, dtype=value.dtype)
        raw = value[:, 2:2+lpe_dim] * scale + mean
        value[:, 2:2+lpe_dim] = (raw * signs[batch["node_batch"]] - mean) / scale
        out[key] = value
    return out
