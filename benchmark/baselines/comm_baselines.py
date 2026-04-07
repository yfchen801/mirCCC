import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import zscore, pearsonr
from typing import Dict, List, Optional, Tuple
import warnings

from .proxy_baselines import compute_proxy_scores, compute_sender_receiver_proxy


class PriorProductBaseline:
    
    def __init__(self, top_k_pairs: int = 10000):
        self.top_k_pairs = top_k_pairs
        self.cellpair_df = None
        self.emit_scores = None
        self.receive_scores = None
    
    def fit_predict(self, adata: ad.AnnData, biogenesis_genes: List[str],
                    risc_genes: List[str], sorting_genes: List[str] = None,
                    gt_labels: pd.DataFrame = None) -> pd.DataFrame:
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        n_cells = adata.n_obs
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        cell_types = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None
        
        emit_scores, risc_scores = compute_sender_receiver_proxy(
            adata, biogenesis_genes, risc_genes, sorting_genes
        )
        
        mirna_targets = adata.uns.get('mirna_targets', {})
        all_targets = set()
        for targets in mirna_targets.values():
            all_targets.update(targets)
        
        target_idx = [gene_to_idx[t] for t in all_targets if t in gene_to_idx]
        
        if len(target_idx) > 0:
            target_expr = X[:, target_idx].mean(axis=1)
            target_min, target_max = target_expr.min(), target_expr.max()
            if target_max > target_min:
                target_norm = (target_expr - target_min) / (target_max - target_min)
            else:
                target_norm = np.zeros_like(target_expr)
            receive_scores = 1.0 - target_norm
            print(f"    [DEBUG Prior] Using target-based receiver score, {len(target_idx)} targets")
        else:
            receive_scores = risc_scores.copy()
            print(f"    [DEBUG Prior] Fallback to RISC-based receiver score (no targets found)")
        
        self.emit_scores = emit_scores.copy()
        self.receive_scores = receive_scores.copy()
        
        emit_min, emit_max = emit_scores.min(), emit_scores.max()
        if emit_max > emit_min:
            emit_norm = (emit_scores - emit_min) / (emit_max - emit_min)
        else:
            emit_norm = np.ones_like(emit_scores) * 0.5
        
        recv_min, recv_max = receive_scores.min(), receive_scores.max()
        if recv_max > recv_min:
            recv_norm = (receive_scores - recv_min) / (recv_max - recv_min)
        else:
            recv_norm = np.ones_like(receive_scores) * 0.5
        
        if gt_labels is not None and len(gt_labels) > 0:
            gt_senders = gt_labels['sender_cell'].values.astype(int)
            gt_receivers = gt_labels['receiver_cell'].values.astype(int)
            gt_scores = emit_norm[gt_senders] * recv_norm[gt_receivers]
            
            records = {
                'sender_cell': gt_senders, 
                'receiver_cell': gt_receivers, 
                'score': gt_scores
            }
            if cell_types is not None:
                records['sender_type'] = cell_types[gt_senders]
                records['receiver_type'] = cell_types[gt_receivers]
            
            self.cellpair_df = pd.DataFrame(records)
            
            print(f"    [DEBUG Prior] Using gt_labels: {len(gt_labels)} pairs")
            print(f"    [DEBUG Prior] emit range: [{emit_norm.min():.4f}, {emit_norm.max():.4f}], "
                  f"recv range: [{recv_norm.min():.4f}, {recv_norm.max():.4f}]")
            print(f"    [DEBUG Prior] score range: [{gt_scores.min():.6f}, {gt_scores.max():.6f}], "
                  f"std: {gt_scores.std():.6f}")
            
        else:
            emit_probs = np.maximum(emit_norm, 1e-10)
            emit_probs = emit_probs / emit_probs.sum()
            recv_probs = np.maximum(recv_norm, 1e-10)
            recv_probs = recv_probs / recv_probs.sum()
            
            n_total = n_cells * n_cells
            if n_total > self.top_k_pairs * 10:
                senders = np.random.choice(n_cells, size=self.top_k_pairs * 2, p=emit_probs)
                receivers = np.random.choice(n_cells, size=self.top_k_pairs * 2, p=recv_probs)
            else:
                senders, receivers = np.meshgrid(np.arange(n_cells), np.arange(n_cells))
                senders = senders.flatten()
                receivers = receivers.flatten()
            
            mask = senders != receivers
            senders = senders[mask]
            receivers = receivers[mask]
            scores = emit_norm[senders] * recv_norm[receivers]
            
            records = {'sender_cell': senders, 'receiver_cell': receivers, 'score': scores}
            if cell_types is not None:
                records['sender_type'] = cell_types[senders]
                records['receiver_type'] = cell_types[receivers]
            
            self.cellpair_df = pd.DataFrame(records)
            self.cellpair_df = self.cellpair_df.nlargest(self.top_k_pairs, 'score')
        
        self.cellpair_df = self.cellpair_df.reset_index(drop=True)
        return self.cellpair_df
    
    def get_celltype_pair_df(self) -> pd.DataFrame:
        if self.cellpair_df is None or 'sender_type' not in self.cellpair_df.columns:
            return pd.DataFrame()
        return self.cellpair_df.groupby(['sender_type', 'receiver_type']).agg({'score': 'sum'}).reset_index()


class PCCCorrelationBaseline:
    
    def __init__(self, n_sample_pairs: int = 5000, proxy_method: str = 'zscore'):
        self.n_sample_pairs = n_sample_pairs
        self.proxy_method = proxy_method
        self.cellpair_df = None
    
    def fit_predict(self, adata: ad.AnnData, mirna_targets: Dict[str, List[str]],
                    biogenesis_genes: Optional[List[str]] = None,
                    risc_genes: Optional[List[str]] = None,
                    gt_labels: pd.DataFrame = None) -> pd.DataFrame:
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        n_cells = X.shape[0]
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        cell_types = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None

        mirna_names = [m for m, tgts in (mirna_targets or {}).items() if tgts]
        
        print(f"    [DEBUG PCC] mirna_targets: {len(mirna_targets or {})} miRNAs, "
              f"non-empty: {len(mirna_names)}")
        
        if len(mirna_names) == 0:
            warnings.warn("mirna_targets is empty -> PCC baseline returns empty.")
            return pd.DataFrame()

        if biogenesis_genes and risc_genes:
            emit_scores, risc_scores = compute_sender_receiver_proxy(adata, biogenesis_genes, risc_genes)
        else:
            emit_scores = np.ones(n_cells)
            risc_scores = np.ones(n_cells)
        
        emit_min, emit_max = emit_scores.min(), emit_scores.max()
        if emit_max > emit_min:
            emit_scores = (emit_scores - emit_min) / (emit_max - emit_min)
        else:
            emit_scores = np.ones(n_cells) * 0.5
        
        target_repression = np.zeros((n_cells, len(mirna_names)))
        for j, mirna in enumerate(mirna_names):
            targets = mirna_targets.get(mirna, [])
            target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if target_idx:
                target_expr = X[:, target_idx].mean(axis=1)
                te_min, te_max = target_expr.min(), target_expr.max()
                if te_max > te_min:
                    te_norm = (target_expr - te_min) / (te_max - te_min)
                else:
                    te_norm = np.zeros(n_cells)
                target_repression[:, j] = 1.0 - te_norm

        if gt_labels is not None and len(gt_labels) > 0:
            senders = gt_labels['sender_cell'].values.astype(int)
            receivers = gt_labels['receiver_cell'].values.astype(int)
            print(f"    [DEBUG PCC] Using gt_labels pairs: {len(senders)}")
        else:
            emit_probs = np.maximum(emit_scores, 1e-10)
            emit_probs = emit_probs / emit_probs.sum()
            risc_probs = np.maximum(risc_scores, 1e-10)
            risc_probs = risc_probs / risc_probs.sum()
            
            senders = np.random.choice(n_cells, size=self.n_sample_pairs, p=emit_probs)
            receivers = np.random.choice(n_cells, size=self.n_sample_pairs, p=risc_probs)
            
            mask = senders != receivers
            senders, receivers = senders[mask], receivers[mask]

        scores = []
        for s, r in zip(senders, receivers):
            sender_ability = emit_scores[s]
            receiver_repress = target_repression[r, :].mean()
            score = sender_ability * receiver_repress
            scores.append(max(0, score))
        
        records = {'sender_cell': senders, 'receiver_cell': receivers, 'score': scores}
        if cell_types is not None:
            records['sender_type'] = cell_types[senders]
            records['receiver_type'] = cell_types[receivers]
        
        self.cellpair_df = pd.DataFrame(records)
        self.cellpair_df = self.cellpair_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        scores_arr = np.array(scores)
        print(f"    [DEBUG PCC] score range: [{scores_arr.min():.6f}, {scores_arr.max():.6f}], "
              f"std: {scores_arr.std():.6f}, n_pairs: {len(self.cellpair_df)}")
        
        return self.cellpair_df
    
    def get_celltype_pair_df(self) -> pd.DataFrame:
        if self.cellpair_df is None or 'sender_type' not in self.cellpair_df.columns:
            return pd.DataFrame()
        return self.cellpair_df.groupby(['sender_type', 'receiver_type']).agg({'score': 'sum'}).reset_index()


def run_prior_product_baseline(adata: ad.AnnData, out_dir: str, top_k: int = 10000,
                                gt_labels: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)
    
    biog = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    sort = adata.uns.get('sorting_genes', [])
    
    print(f"    [DEBUG Prior] biogenesis: {len(biog)}, risc: {len(risc)}, sorting: {len(sort)}")
    
    model = PriorProductBaseline(top_k_pairs=top_k)
    cellpair_df = model.fit_predict(adata, biog, risc, sort, gt_labels)
    celltype_df = model.get_celltype_pair_df()
    
    cellpair_df.to_csv(os.path.join(out_dir, 'prior_product_cellpair.csv'), index=False)
    if len(celltype_df) > 0:
        celltype_df.to_csv(os.path.join(out_dir, 'prior_product_celltype.csv'), index=False)
    
    return cellpair_df, celltype_df


def run_pcc_baseline(adata: ad.AnnData, out_dir: str, n_samples: int = 5000,
                     gt_labels: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)
    
    mirna_targets = adata.uns.get('mirna_targets', {})
    biog = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    
    print(f"    [DEBUG PCC] adata.uns mirna_targets: {len(mirna_targets)} miRNAs")
    
    model = PCCCorrelationBaseline(n_sample_pairs=n_samples)
    cellpair_df = model.fit_predict(adata, mirna_targets, biog, risc, gt_labels)
    celltype_df = model.get_celltype_pair_df()
    
    if len(cellpair_df) > 0:
        cellpair_df.to_csv(os.path.join(out_dir, 'pcc_cellpair.csv'), index=False)
    if len(celltype_df) > 0:
        celltype_df.to_csv(os.path.join(out_dir, 'pcc_celltype.csv'), index=False)
    
    return cellpair_df, celltype_df
