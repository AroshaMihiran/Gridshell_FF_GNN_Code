"""Train one published model configuration on the independently accepted dataset."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from fftgnn.data import load_index, GraphDataset, Normalizer
from fftgnn.experiment import prepare_run
from fftgnn.config import validate_config
from fftgnn.training import train_models
from fftgnn.evaluation import evaluate_registry
from fftgnn.release import ROOT, registry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--model",choices=["M1","M2","M3","M4"],required=True)
    p.add_argument("--output",type=Path,default=ROOT/"runs")
    p.add_argument("--run-name")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--check-data",action="store_true",help="Validate the split, label identities and one graph per split; do not train.")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    record=next(x for x in registry()["models"] if x["model_id"]==args.model)
    cfg=json.loads((ROOT/record["config"]).read_text())
    cfg.update(dataset_root=str(args.dataset.resolve()),output_root=str(args.output.resolve()),
               run_name=args.run_name,resume=args.resume,models=[args.model],seeds=[args.seed],device=args.device)
    validate_config(cfg)
    selected=load_index(args.dataset)
    experiment=json.loads((ROOT/record["experiment"]).read_text())["definition"]
    cols=["sample_id","base_layout_id","normalized_geometry_group","split"]
    actual=selected.sort_values("sample_id")[cols].to_dict("records")
    if actual!=experiment["selection"]:
        raise ValueError("The accepted cases/partitions differ from the published experiment.")
    import hashlib
    for row in selected.itertuples():
        value=hashlib.sha256((Path(row.training_directory)/"label_manifest.json").read_bytes()).hexdigest()
        if value!=experiment["label_manifest_sha256"][row.sample_id]:
            raise ValueError(f"Label manifest differs: {row.sample_id}")
    normalizer=Normalizer.from_dict(json.loads((ROOT/"normalizer.json").read_text()))
    datasets={s:GraphDataset(selected[selected.split==s],normalizer,cfg["lpe_dim"],cfg["cache_size"],True)
              for s in ["train","validation","test"]}
    for dataset in datasets.values():
        dataset[0]
    print("Accepted cases by split:",{s:len(d) for s,d in datasets.items()})
    if args.check_data:
        print("Data checks passed. No training was started.")
        return
    run_dir,fingerprint=prepare_run(cfg,selected,normalizer)
    results=train_models(cfg,datasets,normalizer,run_dir,fingerprint,live=False)
    evaluate_registry(results,datasets,cfg,normalizer,run_dir,fingerprint)
    print("Saved",run_dir)

if __name__=="__main__":
    main()
