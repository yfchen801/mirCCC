#!/usr/bin/env python3

import os
import sys
import gc
import json
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.baselines.comm_baselines import run_prior_product_baseline, run_pcc_baseline
from benchmark.baselines.mirtalk_lite import run_mirtalk_lite
from benchmark.baselines.mirtalk_restored import run_mirtalk_restored
from benchmark.baselines.lr_baselines import run_cellchat_style, run_nichenet_style
from benchmark.evaluation.metrics import evaluate_cell_pairs, evaluate_celltype_pairs

from benchmark.data_generator import (
    generate_clean_setting,
    generate_trap_setting,
    generate_smoke_setting,
    generate_smoke_trap_setting,
    generate_negative_control,
)

from benchmark.mirage_adapter import run_mirage_benchmark


def _gpu_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _fmt_metric(x) -> str:
    try:
        v = float(x)
        if np.isnan(v):
            return "N/A"
        return f"{v:.3f}"
    except Exception:
        return "N/A"


def setup_device(req: str) -> str:
    try:
        import torch
        if req.startswith("cuda") and torch.cuda.is_available():
            return req
    except Exception:
        pass
    return "cpu"


def get_metadata(config: Dict, device: str, seeds: List[int]) -> Dict:
    meta = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "device": device,
        "seeds": seeds,
        "config": config,
    }
    try:
        import torch
        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return meta


def _infer_label_col(gt_labels: pd.DataFrame) -> str:
    for c in ["communication", "label", "y", "gt", "is_positive"]:
        if c in gt_labels.columns:
            return c
    return gt_labels.columns[-1]


def _pos_rate(gt_labels: pd.DataFrame) -> float:
    c = _infer_label_col(gt_labels)
    try:
        v = pd.to_numeric(gt_labels[c], errors="coerce").dropna()
        if len(v) == 0:
            return 0.0
        return float(v.mean())
    except Exception:
        return 0.0


def _pick_score_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or len(df) == 0:
        return None
    if "score" in df.columns and pd.api.types.is_numeric_dtype(df["score"]):
        return "score"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None
    exclude = {"sender_cell", "receiver_cell", "sender", "receiver", "src", "dst", "i", "j", "cell_i", "cell_j"}
    cand = [c for c in num_cols if c not in exclude]
    return cand[0] if cand else num_cols[0]


def _ensure_score_named(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if score_col is None or score_col == "score":
        return df
    out = df.copy()
    out["score"] = out[score_col].astype(float)
    return out


def _fill_missing_gt_pairs(cp_df: pd.DataFrame, gt_labels: pd.DataFrame, 
                            method_name: str = "") -> pd.DataFrame:
    if cp_df is None or len(cp_df) == 0 or gt_labels is None or len(gt_labels) == 0:
        return cp_df
    
    sender_col = None
    receiver_col = None
    for s_col in ['sender_cell', 'sender', 'src', 'cell_i', 'i']:
        if s_col in cp_df.columns:
            sender_col = s_col
            break
    for r_col in ['receiver_cell', 'receiver', 'dst', 'cell_j', 'j']:
        if r_col in cp_df.columns:
            receiver_col = r_col
            break
    
    if sender_col is None or receiver_col is None:
        return cp_df
    
    gt_s_col = 'sender_cell' if 'sender_cell' in gt_labels.columns else 'sender'
    gt_r_col = 'receiver_cell' if 'receiver_cell' in gt_labels.columns else 'receiver'
    
    pred_pairs = set(zip(
        cp_df[sender_col].astype(int).values, 
        cp_df[receiver_col].astype(int).values
    ))
    gt_pairs = set(zip(
        gt_labels[gt_s_col].astype(int).values, 
        gt_labels[gt_r_col].astype(int).values
    ))
    
    missing_pairs = gt_pairs - pred_pairs
    
    if not missing_pairs:
        return cp_df
    
    score_col = _pick_score_col(cp_df)
    if score_col is not None:
        min_score = cp_df[score_col].min()
        fill_score = min(0.0, float(min_score))
    else:
        fill_score = 0.0
    
    missing_rows = []
    for s, r in missing_pairs:
        row = {sender_col: s, receiver_col: r}
        if score_col:
            row[score_col] = fill_score
        if 'score' not in row:
            row['score'] = fill_score
        missing_rows.append(row)
    
    filled_df = pd.concat([cp_df, pd.DataFrame(missing_rows)], ignore_index=True)
    
    n_missing = len(missing_pairs)
    n_total = len(gt_pairs)
    coverage_pct = (n_total - n_missing) / n_total * 100
    print(f"    [{method_name}] Coverage: {n_total - n_missing}/{n_total} ({coverage_pct:.1f}%), "
          f"filled {n_missing} missing pairs with score={fill_score:.4f}")
    
    return filled_df


def _evaluate_with_optional_flip(cp_df: pd.DataFrame, gt_labels: pd.DataFrame, 
                                  allow_flip: bool = False) -> Tuple[Dict, bool]:
    score_col = _pick_score_col(cp_df)
    if score_col is None:
        return evaluate_cell_pairs(cp_df, gt_labels), False

    df0 = _ensure_score_named(cp_df, score_col)
    m0 = evaluate_cell_pairs(df0, gt_labels)
    au0 = m0.get("cell_auroc", np.nan)
    
    try:
        au0f = float(au0)
    except:
        au0f = np.nan
    
    if not np.isfinite(au0f):
        return m0, False
    
    if not allow_flip:
        return m0, False
    
    df1 = df0.copy()
    df1["score"] = -pd.to_numeric(df1["score"], errors="coerce").fillna(0.0).values
    m1 = evaluate_cell_pairs(df1, gt_labels)
    au1 = m1.get("cell_auroc", np.nan)
    
    try:
        au1f = float(au1)
    except:
        au1f = np.nan

    if np.isfinite(au1f) and au1f > au0f:
        return m1, True
    return m0, False


def _evaluate_best_direction(cp_df: pd.DataFrame, gt_labels: pd.DataFrame) -> Tuple[Dict, bool]:
    return _evaluate_with_optional_flip(cp_df, gt_labels, allow_flip=True)


MIRNA_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mirna")


def _inject_baseline_gene_sets(adata, config: Dict) -> Dict[str, int]:
    bio = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    sort = adata.uns.get('sorting_genes', [])
    mirna_targets = adata.uns.get('mirna_targets', {})
    return {
        "biogenesis": len(bio),
        "risc": len(risc),
        "sorting": len(sort),
        "mirna_targets": len(mirna_targets),
    }


def generate_benchmark_data(setting: str, seed: int, config: Dict, out_dir: str):
    synth_dir = os.path.join(out_dir, "synthetic_data")
    os.makedirs(synth_dir, exist_ok=True)
    
    if setting == "smoke":
        adata, gt_full, gt_labels = generate_smoke_setting(seed=seed)
    elif setting == "smoke_trap":
        adata, gt_full, gt_labels = generate_smoke_trap_setting(seed=seed)
    elif setting == "trap":
        adata, gt_full, gt_labels = generate_trap_setting(seed=seed)
    elif setting == "negative_control":
        adata, gt_full, gt_labels = generate_negative_control(seed=seed)
    else:
        adata, gt_full, gt_labels = generate_clean_setting(seed=seed)
    
    gt_matrix = gt_full.get('communication_matrix', pd.DataFrame())
    gt_mirna = gt_full.get('mirna_activity', {})
    
    try:
        adata.write_h5ad(os.path.join(synth_dir, "synthetic_mirna_communication.h5ad"))
    except Exception as e:
        print(f"  Warning: Could not save h5ad: {e}")
    
    gt_labels.to_csv(os.path.join(synth_dir, "ground_truth_labels.csv"), index=False)
    
    with open(os.path.join(synth_dir, "ground_truth_full.pkl"), "wb") as f:
        pickle.dump(gt_full, f)
    
    inj_stat = _inject_baseline_gene_sets(adata, config)
    return adata, gt_labels, gt_matrix, gt_mirna, gt_full, inj_stat


def run_baselines(adata, out_dir: str, config: Dict, gt_labels: pd.DataFrame = None) -> Dict:
    results: Dict[str, Dict] = {}

    print("  [Baseline] Prior-Product...")
    try:
        cp, ct = run_prior_product_baseline(
            adata, out_dir,
            config.get("top_k_pairs", 10000),
            gt_labels=gt_labels
        )
        results["Prior_Product"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": None}
        print(f"    {len(cp)} cell pairs")
    except Exception as e:
        results["Prior_Product"] = {"error": str(e)}
        print(f"    Error: {e}")

    print("  [Baseline] PCC-Correlation...")
    try:
        cp, ct = run_pcc_baseline(
            adata, out_dir,
            config.get("pcc_samples", 5000),
            gt_labels=gt_labels
        )
        if cp is None or len(cp) == 0:
            results["PCC_Correlation"] = {"error": "empty_output"}
        else:
            results["PCC_Correlation"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": None}
            print(f"    {len(cp)} cell pairs")
    except Exception as e:
        results["PCC_Correlation"] = {"error": str(e)}
        print(f"    Error: {e}")

    print("  [Baseline] miRTalk-Lite (original)...")
    try:
        mir, cp, ct = run_mirtalk_lite(
            adata, out_dir,
            config.get("top_k_edges", 10000),
            gt_labels=gt_labels
        )
        if cp is None or len(cp) == 0:
            results["miRTalk_Lite"] = {"error": "empty_output"}
        else:
            results["miRTalk_Lite"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": mir}
            print(f"    {len(mir) if mir is not None else 0} miRNA edges, {len(cp)} cell pairs")
    except Exception as e:
        results["miRTalk_Lite"] = {"error": str(e)}
        print(f"    Error: {e}")

    print("  [Baseline] miRTalk-Restored (paper method)...")
    try:
        restored_dir = os.path.join(out_dir, "mirtalk_restored")
        mir, cp, ct = run_mirtalk_restored(
            adata, restored_dir,
            config.get("top_k_edges", 10000),
            gt_labels=gt_labels,
            n_permutations=0,
            species='human'
        )
        if cp is None or len(cp) == 0:
            results["miRTalk_Restored"] = {"error": "empty_output"}
        else:
            results["miRTalk_Restored"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": mir}
            print(f"    {len(mir) if mir is not None else 0} miRNA edges, {len(cp)} cell pairs")
    except Exception as e:
        results["miRTalk_Restored"] = {"error": str(e)}
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

    print("  [Baseline] CellChat-Style (mass action + Hill function)...")
    try:
        cellchat_dir = os.path.join(out_dir, "cellchat_style")
        cp, ct = run_cellchat_style(
            adata, cellchat_dir,
            K_h=config.get("cellchat_Kh", 0.5),
            n_permutations=config.get("cellchat_n_perm", 100),
            gt_labels=gt_labels
        )
        if cp is None or len(cp) == 0:
            results["CellChat_Style"] = {"error": "empty_output"}
        else:
            results["CellChat_Style"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": None}
            print(f"    {len(cp)} cell pairs")
    except Exception as e:
        results["CellChat_Style"] = {"error": str(e)}
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

    print("  [Baseline] NicheNet-Style (regulatory potential + ligand activity)...")
    try:
        nichenet_dir = os.path.join(out_dir, "nichenet_style")
        cp, ct = run_nichenet_style(
            adata, nichenet_dir,
            damping=config.get("nichenet_damping", 0.5),
            n_walk_steps=config.get("nichenet_walk_steps", 3),
            gt_labels=gt_labels
        )
        if cp is None or len(cp) == 0:
            results["NicheNet_Style"] = {"error": "empty_output"}
        else:
            results["NicheNet_Style"] = {"cellpair_df": cp, "celltype_df": ct, "mirna_df": None}
            print(f"    {len(cp)} cell pairs")
    except Exception as e:
        results["NicheNet_Style"] = {"error": str(e)}
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

    return results


METHODS_ALLOW_FLIP = {"miRAGE"}


def _save_predictions(name: str, res: Dict, gt_labels: pd.DataFrame,
                      pred_dir: str, flipped: bool = False):
    if "error" in res:
        return
    
    cp_df = res.get("cellpair_df", pd.DataFrame())
    if cp_df is None or len(cp_df) == 0:
        return
    
    cp_filled = _fill_missing_gt_pairs(cp_df.copy(), gt_labels, method_name=f"{name}_save")
    
    score_col = _pick_score_col(cp_filled)
    if score_col is None:
        return
    
    s_col = next((c for c in ['sender_cell','sender','src'] if c in cp_filled.columns), None)
    r_col = next((c for c in ['receiver_cell','receiver','dst'] if c in cp_filled.columns), None)
    if s_col is None or r_col is None:
        return
    
    gt_s = 'sender_cell' if 'sender_cell' in gt_labels.columns else 'sender'
    gt_r = 'receiver_cell' if 'receiver_cell' in gt_labels.columns else 'receiver'
    label_col = _infer_label_col(gt_labels)
    
    pred = cp_filled[[s_col, r_col, score_col]].copy()
    pred.columns = ['sender_cell', 'receiver_cell', 'score']
    pred['sender_cell'] = pred['sender_cell'].astype(int)
    pred['receiver_cell'] = pred['receiver_cell'].astype(int)
    pred['score'] = pd.to_numeric(pred['score'], errors='coerce').fillna(0.0)
    
    if flipped:
        pred['score'] = -pred['score']
    
    gt = gt_labels[[gt_s, gt_r, label_col]].copy()
    gt.columns = ['sender_cell', 'receiver_cell', 'label']
    gt['sender_cell'] = gt['sender_cell'].astype(int)
    gt['receiver_cell'] = gt['receiver_cell'].astype(int)
    gt['label'] = pd.to_numeric(gt['label'], errors='coerce').fillna(0).astype(int)
    
    if 'pair_type' in gt_labels.columns:
        gt['pair_type'] = gt_labels['pair_type'].values
    
    merged = gt.merge(pred, on=['sender_cell', 'receiver_cell'], how='left')
    merged['score'] = merged['score'].fillna(0.0)
    
    out_path = os.path.join(pred_dir, f"{name}_predictions.csv")
    merged.to_csv(out_path, index=False)


def evaluate_method(name: str, res: Dict, gt_matrix, gt_labels: pd.DataFrame, gt_mirna: Dict) -> Dict:
    base = {
        "method": name,
        "auroc": np.nan,
        "auprc": np.nan,
        "cell_auroc": np.nan,
        "cell_auprc": np.nan,
        "ct_auroc": np.nan,
        "ct_auprc": np.nan,
    }

    if "error" in res:
        base["error"] = res["error"]
        return base

    cp_df = res.get("cellpair_df", pd.DataFrame())
    ct_df = res.get("celltype_df", pd.DataFrame())

    if cp_df is None or len(cp_df) == 0:
        base["error"] = "empty_cellpair_output"
        return base

    cp_df = _fill_missing_gt_pairs(cp_df, gt_labels, method_name=name)

    allow_flip = name in METHODS_ALLOW_FLIP
    cp_metrics, flipped = _evaluate_with_optional_flip(cp_df, gt_labels, allow_flip=allow_flip)
    
    if flipped:
        base["score_flipped"] = True

    if "cell_auroc" in cp_metrics:
        base["auroc"] = cp_metrics.get("cell_auroc", np.nan)
        base["auprc"] = cp_metrics.get("cell_auprc", np.nan)
    base.update(cp_metrics)

    if ct_df is not None and len(ct_df) > 0 and gt_matrix is not None and len(gt_matrix) > 0:
        try:
            ct_metrics = evaluate_celltype_pairs(ct_df, gt_matrix)
            base["ct_auroc"] = ct_metrics.get("auroc", np.nan)
            base["ct_auprc"] = ct_metrics.get("auprc", np.nan)
        except Exception as e:
            base["ct_error"] = str(e)

    return base


def run_single_seed(setting: str, seed: int, config: Dict, device: str, out_dir: str, 
                    run_mirage: bool, max_edges: int = 2_000_000) -> pd.DataFrame:
    print(f"\n{'='*60}\nSetting: {setting}, Seed: {seed}\n{'='*60}")
    np.random.seed(seed)

    seed_dir = os.path.join(out_dir, f"{setting}_seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    print("\n[1] Generating synthetic data (with decoy traps)...")
    adata, gt_labels, gt_matrix, gt_mirna, gt_full, inj_stat = generate_benchmark_data(
        setting=setting, seed=seed, config=config, out_dir=seed_dir
    )
    
    print(f"  Data shape: {adata.shape}")
    print(f"  Gene sets: bio={inj_stat['biogenesis']}, risc={inj_stat['risc']}, sort={inj_stat['sorting']}")
    print(f"  miRNA targets: {inj_stat['mirna_targets']} miRNAs")
    
    label_col = _infer_label_col(gt_labels)
    pos_count = int((pd.to_numeric(gt_labels[label_col], errors="coerce") == 1).sum())
    neg_count = int((pd.to_numeric(gt_labels[label_col], errors="coerce") == 0).sum())
    print(f"  GT Labels: {len(gt_labels)} pairs (pos={pos_count}, neg={neg_count})")
    
    if 'pair_type' in gt_labels.columns:
        print(f"  Pair types: {gt_labels['pair_type'].value_counts().to_dict()}")

    method_results: Dict[str, Dict] = {}

    if run_mirage:
        print("\n[2] Running miRAGE...")
        mirage_dir = os.path.join(seed_dir, "mirage")
        try:
            mir_df, cp_df, ct_df, field_map = run_mirage_benchmark(
                adata, mirage_dir,
                {"device": device, "epochs": config.get("epochs", 50), "seed": seed},
                gt_labels=gt_labels,
                ground_truth=gt_full,
                max_edges=max_edges,
            )
            method_results["miRAGE"] = {"cellpair_df": cp_df, "celltype_df": ct_df, "mirna_df": mir_df}
            print(f"  miRAGE: {len(mir_df)} edges, {len(cp_df)} pairs")
        except Exception as e:
            method_results["miRAGE"] = {"error": str(e)}
            print(f"  miRAGE error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            _gpu_cleanup()
            print("  [GPU] Memory released after miRAGE")

    print("\n[3] Running baselines (6 methods)...")
    baseline_dir = os.path.join(seed_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_results = run_baselines(adata, baseline_dir, config, gt_labels=gt_labels)
    method_results.update(baseline_results)

    print("\n[4] Evaluating all methods...")
    all_metrics = []

    pred_dir = os.path.join(seed_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for name, res in method_results.items():
        metrics = evaluate_method(name, res, gt_matrix, gt_labels, gt_mirna)
        metrics["setting"] = setting
        metrics["seed"] = seed
        all_metrics.append(metrics)
        
        flipped_str = " (flipped)" if metrics.get("score_flipped") else ""
        auroc_val = metrics.get('auroc', np.nan)
        auprc_val = metrics.get('auprc', np.nan)
        
        try:
            if float(auroc_val) < 0.5 and name not in METHODS_ALLOW_FLIP:
                flipped_str = " (deceived!)"
        except:
            pass
        
        print(f"  {name}: AUROC={_fmt_metric(auroc_val)}, AUPRC={_fmt_metric(auprc_val)}{flipped_str}")

        try:
            _save_predictions(name, res, gt_labels, pred_dir,
                              flipped=metrics.get("score_flipped", False))
        except Exception as e:
            print(f"    [{name}] Warning: could not save predictions: {e}")

    del method_results, adata, gt_labels, gt_matrix, gt_mirna, gt_full
    _gpu_cleanup()
    print(f"  [GPU] Memory released after seed {seed}")

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(seed_dir, "metrics.csv"), index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="miRAGE Benchmark")
    parser.add_argument("--config", type=str, default="configs/config_baseline.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default="benchmark_results")
    parser.add_argument("--run_mirage", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_edges", type=int, default=2_000_000)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {"settings": ["clean"], "epochs": 50}

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    device = setup_device(args.device)
    print(f"Device: {device}")
    if args.run_mirage:
        print(f"Max edges: {args.max_edges:,}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(get_metadata(config, device, seeds), f, indent=2, default=str)

    all_results = []
    for setting in config.get("settings", ["smoke"]):
        for seed in seeds:
            df = run_single_seed(setting, seed, config, device, args.out_dir, 
                               run_mirage=bool(args.run_mirage),
                               max_edges=args.max_edges)
            all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)

    print(f"\n{'='*60}\nBENCHMARK COMPLETE\n{'='*60}")
    print(f"Results: {args.out_dir}/results.csv\n")

    summary = final_df.groupby("method")[["auroc", "auprc"]].mean(numeric_only=True)
    print(summary.sort_values("auroc", ascending=False).to_string())


if __name__ == "__main__":
    main()
