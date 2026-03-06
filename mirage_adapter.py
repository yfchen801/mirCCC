
import os
import sys
import pandas as pd
import anndata as ad
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from miRAGE_integrated_pipeline_api import (
        Config as MirageConfig, run_full_pipeline, setup_output_dirs, TORCH_AVAILABLE
    )
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("Could not import miRAGE pipeline api.")

MIRNA_DATA_DIR = "/home/YifanChen/miRAGE/mirCCC/mirna"
MIR2TAR_PATH = f"{MIRNA_DATA_DIR}/mir2tar.csv"
BIOGENESIS_PATH = f"{MIRNA_DATA_DIR}/biogenesis.csv"
RISC_PATH = f"{MIRNA_DATA_DIR}/risc.csv"
SORTING_PATH = f"{MIRNA_DATA_DIR}/sorting.csv"
GENEINFO_PATH = f"{MIRNA_DATA_DIR}/geneinfo.csv"


def estimate_graph_size(n_mirnas: int, top_s: int, top_l: int) -> int:
    return n_mirnas * top_s * top_l


def get_optimal_params(n_cells: int, n_mirnas: int, max_edges: int = 2_000_000) -> Dict:
    top_s = 500
    top_r = 500
    top_l = 20
    
    estimated_edges = estimate_graph_size(n_mirnas, top_s, top_l)
    
    if estimated_edges <= max_edges:
        return {
            'TOP_S_SENDERS': top_s,
            'TOP_R_RECEIVERS': top_r,
            'TOP_L_PER_SENDER': top_l,
            'MAX_MIRNAS': n_mirnas,
            'estimated_edges': estimated_edges,
        }
    
    
    target_mirnas = max_edges // (top_s * top_l)
    if target_mirnas >= 30:
        return {
            'TOP_S_SENDERS': top_s,
            'TOP_R_RECEIVERS': top_r,
            'TOP_L_PER_SENDER': top_l,
            'MAX_MIRNAS': target_mirnas,
            'estimated_edges': estimate_graph_size(target_mirnas, top_s, top_l),
        }
    
    target_mirnas = 50
    target_edges_per_mirna = max_edges // target_mirnas
    
    top_l = max(5, min(20, int(np.sqrt(target_edges_per_mirna / 25))))
    top_s = min(n_cells, target_edges_per_mirna // top_l)
    top_r = top_s
    
    return {
        'TOP_S_SENDERS': top_s,
        'TOP_R_RECEIVERS': top_r,
        'TOP_L_PER_SENDER': top_l,
        'MAX_MIRNAS': target_mirnas,
        'estimated_edges': estimate_graph_size(target_mirnas, top_s, top_l),
    }


def run_mirage_pipeline(adata: ad.AnnData, out_dir: str, config: Dict,
                        gt_labels: pd.DataFrame = None, 
                        ground_truth: Dict = None,
                        max_edges: int = 2_000_000,
                        force_params: Dict = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = MirageConfig()
    cfg.OUTPUT_DIR = out_dir
    cfg.MIR2TAR_PATH = MIR2TAR_PATH
    cfg.BIOGENESIS_PATH = BIOGENESIS_PATH
    cfg.RISC_PATH = RISC_PATH
    cfg.SORTING_PATH = SORTING_PATH
    cfg.GENEINFO_PATH = GENEINFO_PATH
    
    cfg.DEVICE = config.get('device', 'cpu')
    cfg.NUM_EPOCHS = config.get('epochs', 200)
    cfg.USE_EXISTING_DATA = True
    cfg.EXISTING_DATA_PATH = os.path.join(out_dir, "synthetic_data", "synthetic_mirna_communication.h5ad")
    
    try:
        mir2tar = pd.read_csv(MIR2TAR_PATH, index_col=0)
        n_mirnas = len(mir2tar)
    except:
        n_mirnas = 1000
    
    if force_params:
        opt_params = force_params
    else:
        opt_params = get_optimal_params(
            n_cells=adata.n_obs,
            n_mirnas=n_mirnas,
            max_edges=max_edges
        )
    
    cfg.TOP_S_SENDERS = opt_params.get('TOP_S_SENDERS', 500)
    cfg.TOP_R_RECEIVERS = opt_params.get('TOP_R_RECEIVERS', 500)
    cfg.TOP_L_PER_SENDER = opt_params.get('TOP_L_PER_SENDER', 20)
    
    mirna_list = adata.uns.get('mirna_list', [])
    if mirna_list:
        print(f"    Using {len(mirna_list)} miRNAs from data generator")
        limited_mir2tar_path = os.path.join(out_dir, "mir2tar_limited.csv")
        _create_mir2tar_from_list(MIR2TAR_PATH, limited_mir2tar_path, mirna_list)
        cfg.MIR2TAR_PATH = limited_mir2tar_path
    elif opt_params.get('MAX_MIRNAS', n_mirnas) < n_mirnas:
        limited_mir2tar_path = os.path.join(out_dir, "mir2tar_limited.csv")
        _create_limited_mir2tar(MIR2TAR_PATH, limited_mir2tar_path, opt_params['MAX_MIRNAS'], adata)
        cfg.MIR2TAR_PATH = limited_mir2tar_path
    
    print(f"\n  [miRAGE GPU-Optimized] Parameters:")
    print(f"    TOP_S_SENDERS: {cfg.TOP_S_SENDERS}")
    print(f"    TOP_R_RECEIVERS: {cfg.TOP_R_RECEIVERS}")
    print(f"    TOP_L_PER_SENDER: {cfg.TOP_L_PER_SENDER}")
    print(f"    MAX_MIRNAS: {len(mirna_list) if mirna_list else opt_params.get('MAX_MIRNAS', 'auto')}")
    print(f"    Estimated edges: {opt_params.get('estimated_edges', 'N/A'):,}")
    print(f"    Device: {cfg.DEVICE}")
    
    setup_output_dirs(cfg)
    
    synth_path = os.path.join(out_dir, "synthetic_data")
    os.makedirs(synth_path, exist_ok=True)
    adata.write_h5ad(os.path.join(synth_path, "synthetic_mirna_communication.h5ad"))
    
    if gt_labels is not None:
        gt_labels.to_csv(os.path.join(synth_path, "ground_truth_labels.csv"), index=False)
        gt_labels.to_csv(os.path.join(out_dir, "ground_truth_labels.csv"), index=False)
    
    if ground_truth is not None:
        import pickle
        gt_full_path = os.path.join(synth_path, "ground_truth_full.pkl")
        with open(gt_full_path, 'wb') as f:
            pickle.dump(ground_truth, f)
        
        n_true_s = len(ground_truth.get('true_sender_cells', []))
        n_true_r = len(ground_truth.get('true_receiver_cells', []))
        print(f"    Active cells: {n_true_s} senders, {n_true_r} receivers")
        
        if n_true_s < cfg.TOP_S_SENDERS and n_true_r < cfg.TOP_R_RECEIVERS:
            print(f"    âœ“ Active cells should be covered by graph")
        else:
            print(f"    âš  Active cells may exceed graph coverage!")
    
    try:
        results = run_full_pipeline(cfg)
        results['mode'] = 'real'
    except Exception as e:
        print(f"  [miRAGE] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results = {'mode': 'failed'}
        
    return results


def _create_limited_mir2tar(original_path: str, output_path: str, max_mirnas: int, adata: ad.AnnData):
    df = pd.read_csv(original_path)
    
    if 'miRNA' not in df.columns or 'target_gene' not in df.columns:
        print(f"    Warning: mir2tar.csv format unexpected, columns: {df.columns.tolist()}")
        import shutil
        shutil.copy(original_path, output_path)
        return
    
    var_genes = set(adata.var_names.astype(str))
    
    mirna_scores = {}
    for mirna, group in df.groupby('miRNA'):
        targets = group['target_gene'].dropna().unique()
        targets_in_data = sum(1 for t in targets if t in var_genes)
        mirna_scores[str(mirna)] = targets_in_data
    
    sorted_mirnas = sorted(mirna_scores.keys(), key=lambda m: mirna_scores[m], reverse=True)
    selected_mirnas = sorted_mirnas[:max_mirnas]
    
    limited_df = df[df['miRNA'].isin(selected_mirnas)]
    limited_df.to_csv(output_path, index=False)
    
    print(f"    Created limited mir2tar: {len(selected_mirnas)} miRNAs")
    if selected_mirnas:
        print(f"    Top miRNA target counts: {[mirna_scores[m] for m in selected_mirnas[:5]]}")

def _create_mir2tar_from_list(original_path: str, output_path: str, mirna_list: list[str]):
    df = pd.read_csv(original_path)
    
    limited_df = df[df['miRNA'].isin(mirna_list)]
    limited_df.to_csv(output_path, index=False)
    
    print(f"    Created mir2tar with {len(mirna_list)} specified miRNAs, {len(limited_df)} records")
    
class MirageOutputAdapter:
    
    def __init__(self, results: Dict, adata: ad.AnnData, 
                 gt_labels: pd.DataFrame = None, out_dir: str = None):
        self.results = results
        self.adata = adata
        self.gt_labels = gt_labels
        self.out_dir = out_dir
        self.field_mapping = {'mode': results.get('mode', 'unknown')}
        self._cellpair_df = None
        self._mirna_edge_df = None
    
    def get_mirna_edge_df(self) -> pd.DataFrame:
        if self._mirna_edge_df is not None: 
            return self._mirna_edge_df
        if self.out_dir:
            p = os.path.join(self.out_dir, 'results', 'mirage_mirna_edges.csv')
            if os.path.exists(p):
                self._mirna_edge_df = pd.read_csv(p)
                return self._mirna_edge_df
        return pd.DataFrame()
    
    def _rescore_from_edges(self) -> pd.DataFrame:
        edge_df = self.get_mirna_edge_df()
        if len(edge_df) == 0:
            return pd.DataFrame()
        
        has_proxy = 'mir_proxy_01' in edge_df.columns
        has_repress = 'repress_01' in edge_df.columns
        has_emit = 'emit_prior' in edge_df.columns
        has_risc = 'risc_01' in edge_df.columns
        
        if has_proxy and has_repress:
            proxy = edge_df['mir_proxy_01'].fillna(0).values
            repress = edge_df['repress_01'].fillna(0).values
            
            match = proxy * repress
            
            if has_emit and has_risc:
                emit = edge_df['emit_prior'].fillna(0).values
                risc = edge_df['risc_01'].fillna(0).values
                gate = np.sqrt(np.maximum(emit, 0) * np.maximum(risc, 0))
                edge_df['rescore'] = match * gate
            else:
                edge_df['rescore'] = match
        elif 'score' in edge_df.columns:
            edge_df['rescore'] = edge_df['score'].fillna(0).values
        else:
            return pd.DataFrame()
        
        group_cols = ['sender_cell', 'receiver_cell']
        agg_df = edge_df.groupby(group_cols).agg(
            score=('rescore', 'max'),
            n_mirnas=('mirna', 'nunique'),
        ).reset_index()
        
        agg_df = agg_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        pos_scores = agg_df.loc[agg_df['score'] > 0, 'score']
        if len(pos_scores) > 0:
            print(f"  [Adapter-rescore] {len(agg_df)} pairs, "
                  f"score>0: {len(pos_scores)}, "
                  f"median={pos_scores.median():.6f}, "
                  f"max={pos_scores.max():.6f}")
        
        return agg_df
    
    def get_cellpair_df(self) -> pd.DataFrame:
        if self._cellpair_df is not None: 
            return self._cellpair_df
        
        pred_df = self._rescore_from_edges()
        
        if len(pred_df) == 0 and self.out_dir:
            for fname in ['mirage_cellpair_max.csv', 'mirage_cellpair_topk_sum.csv', 'mirage_cellpair_sum.csv']:
                p = os.path.join(self.out_dir, 'results', fname)
                if os.path.exists(p):
                    pred_df = pd.read_csv(p)
                    print(f"  [Adapter] Fallback: loaded {fname}")
                    break
        
        if len(pred_df) == 0:
            print("  [Adapter] Warning: No predictions found.")
            return pd.DataFrame()

        print(f"  [Adapter] Loaded {len(pred_df)} predicted pairs")

        if self.gt_labels is not None:
            gt = self.gt_labels[['sender_cell', 'receiver_cell']].copy()
            gt['sender_cell'] = gt['sender_cell'].astype(int)
            gt['receiver_cell'] = gt['receiver_cell'].astype(int)
            pred_df['sender_cell'] = pred_df['sender_cell'].astype(int)
            pred_df['receiver_cell'] = pred_df['receiver_cell'].astype(int)
            
            merged = gt.merge(pred_df, on=['sender_cell', 'receiver_cell'], how='left')
            merged['score'] = merged['score'].fillna(0.0)
            
            if 'cell_type' in self.adata.obs.columns:
                cts = self.adata.obs['cell_type'].values
                merged['sender_type'] = [cts[i] for i in merged['sender_cell']]
                merged['receiver_type'] = [cts[i] for i in merged['receiver_cell']]
                
            self._cellpair_df = merged
            
            n_hit = (merged['score'] > 0).sum()
            n_total = len(merged)
            coverage = n_hit / n_total * 100
            print(f"  [Adapter] Coverage: {n_hit}/{n_total} ({coverage:.1f}%)")
            
            if 'pair_type' in self.gt_labels.columns:
                merged_full = merged.merge(
                    self.gt_labels[['sender_cell', 'receiver_cell', 'communication', 'pair_type']], 
                    on=['sender_cell', 'receiver_cell'], 
                    how='left'
                )
                
                print(f"\n  [Adapter] Score by pair type:")
                for pt in sorted(merged_full['pair_type'].dropna().unique()):
                    mask = merged_full['pair_type'] == pt
                    scores = merged_full.loc[mask, 'score']
                    covered = (scores > 0).sum()
                    print(f"    {pt:20s}: n={mask.sum()}, covered={covered}, "
                          f"mean={scores.mean():.6f}, median={scores[scores>0].median() if covered > 0 else 0:.6f}")
        else:
            self._cellpair_df = pred_df
            
        return self._cellpair_df
    
    def get_celltype_pair_df(self) -> pd.DataFrame:
        cp = self.get_cellpair_df()
        if len(cp) > 0 and 'sender_type' in cp.columns:
            return cp.groupby(['sender_type', 'receiver_type'])['score'].sum().reset_index()
        return pd.DataFrame()


def run_mirage_benchmark(adata, out_dir, config, gt_labels=None, ground_truth=None,
                        max_edges: int = 2_000_000):
    results = run_mirage_pipeline(
        adata, out_dir, config, gt_labels, ground_truth,
        max_edges=max_edges
    )
    adapter = MirageOutputAdapter(results, adata, gt_labels, out_dir)
    
    return (adapter.get_mirna_edge_df(), 
            adapter.get_cellpair_df(), 
            adapter.get_celltype_pair_df(), 
            adapter.field_mapping)


GPU_CONFIGS = {
    '4GB': {'max_edges': 500_000, 'estimated_memory': '1-2 GB'},
    '8GB': {'max_edges': 1_500_000, 'estimated_memory': '3-5 GB'},
    '12GB': {'max_edges': 3_000_000, 'estimated_memory': '5-8 GB'},
    '16GB': {'max_edges': 5_000_000, 'estimated_memory': '8-12 GB'},
    '24GB': {'max_edges': 8_000_000, 'estimated_memory': '12-18 GB'},
}


def get_config_for_gpu(gpu_memory_gb: int) -> Dict:
    if gpu_memory_gb <= 4:
        return GPU_CONFIGS['4GB']
    elif gpu_memory_gb <= 8:
        return GPU_CONFIGS['8GB']
    elif gpu_memory_gb <= 12:
        return GPU_CONFIGS['12GB']
    elif gpu_memory_gb <= 16:
        return GPU_CONFIGS['16GB']
    else:
        return GPU_CONFIGS['24GB']
