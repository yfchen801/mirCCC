import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import zscore
from typing import Dict, List, Optional, Tuple


class GeneSetZscoreProxy:
    
    def __init__(self, min_targets: int = 3):
        self.min_targets = min_targets
        self.proxy_matrix = None
        self.mirna_names = None
    
    def fit_transform(self, adata: ad.AnnData, mirna_targets: Dict[str, List[str]]) -> np.ndarray:
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        n_cells = X.shape[0]
        X_zscore = zscore(X, axis=0, nan_policy='omit')
        X_zscore = np.nan_to_num(X_zscore, nan=0.0)
        
        valid_mirnas = []
        proxy_scores = []
        
        for mirna, targets in mirna_targets.items():
            target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(target_idx) < self.min_targets:
                continue
            target_zscore = X_zscore[:, target_idx]
            proxy = -np.mean(target_zscore, axis=1)
            valid_mirnas.append(mirna)
            proxy_scores.append(proxy)
        
        if not valid_mirnas:
            self.proxy_matrix = np.zeros((n_cells, 0))
            self.mirna_names = []
            return self.proxy_matrix
        
        self.proxy_matrix = np.column_stack(proxy_scores)
        self.mirna_names = valid_mirnas
        return self.proxy_matrix


class GeneSetAUCellProxy:
    
    def __init__(self, min_targets: int = 3, top_percent: float = 0.2):
        self.min_targets = min_targets
        self.top_percent = top_percent
        self.proxy_matrix = None
        self.mirna_names = None
    
    def fit_transform(self, adata: ad.AnnData, mirna_targets: Dict[str, List[str]]) -> np.ndarray:
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        n_cells, n_genes = X.shape
        ranks = np.zeros_like(X)
        for i in range(n_cells):
            ranks[i] = np.argsort(np.argsort(X[i]))
        
        bottom_k = int(n_genes * self.top_percent)
        
        valid_mirnas = []
        proxy_scores = []
        
        for mirna, targets in mirna_targets.items():
            target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(target_idx) < self.min_targets:
                continue
            
            scores = []
            for i in range(n_cells):
                target_ranks = ranks[i, target_idx]
                in_bottom = np.sum(target_ranks < bottom_k)
                scores.append(in_bottom / len(target_idx))
            
            valid_mirnas.append(mirna)
            proxy_scores.append(np.array(scores))
        
        if not valid_mirnas:
            self.proxy_matrix = np.zeros((n_cells, 0))
            self.mirna_names = []
            return self.proxy_matrix
        
        self.proxy_matrix = np.column_stack(proxy_scores)
        self.mirna_names = valid_mirnas
        return self.proxy_matrix


def compute_proxy_scores(adata: ad.AnnData, mirna_targets: Dict[str, List[str]], 
                         method: str = 'zscore') -> Tuple[np.ndarray, List[str]]:
    if method == 'zscore':
        proxy = GeneSetZscoreProxy()
    elif method == 'aucell':
        proxy = GeneSetAUCellProxy()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    proxy.fit_transform(adata, mirna_targets)
    return proxy.proxy_matrix, proxy.mirna_names


def compute_sender_receiver_proxy(adata: ad.AnnData, biogenesis_genes: List[str],
                                   risc_genes: List[str], sorting_genes: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_cells = X.shape[0]
    
    biog_idx = [gene_to_idx[g] for g in biogenesis_genes if g in gene_to_idx]
    risc_idx = [gene_to_idx[g] for g in risc_genes if g in gene_to_idx]
    sort_idx = [gene_to_idx[g] for g in (sorting_genes or []) if g in gene_to_idx]
    
    emit_score = np.zeros(n_cells)
    if biog_idx:
        emit_score += X[:, biog_idx].mean(axis=1)
    if sort_idx:
        emit_score += X[:, sort_idx].mean(axis=1)
    
    risc_score = np.zeros(n_cells)
    if risc_idx:
        risc_score = X[:, risc_idx].mean(axis=1)
    
    return emit_score, risc_score
