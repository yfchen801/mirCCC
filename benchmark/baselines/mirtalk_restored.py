import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import zscore
from typing import Dict, List, Optional, Tuple
import os
import warnings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


EVBS_GENES_HUMAN = [
    'ALIX', 'TSG101', 'CD63', 'CD81', 'CD9', 'RAB27A', 'RAB27B', 'RAB11A', 
    'RAB11B', 'RAB35', 'SMPD3', 'VPS4A', 'VPS4B', 'CHMP4A', 'CHMP4B', 'CHMP4C',
    'SDCBP', 'PDCD6IP', 'VTA1', 'STAM', 'HGS', 'MVB12A', 'MVB12B', 'VPS28',
    'VPS37A', 'VPS37B', 'VPS37C', 'VPS37D', 'CHMP1A', 'CHMP1B', 'CHMP2A', 
    'CHMP2B', 'CHMP3', 'CHMP5', 'CHMP6', 'CHMP7', 'IST1'
]

EVBS_GENES_MOUSE = [
    'Alix', 'Tsg101', 'Cd63', 'Cd81', 'Cd9', 'Rab27a', 'Rab27b', 'Rab11a',
    'Rab11b', 'Rab35', 'Smpd3', 'Vps4a', 'Vps4b', 'Chmp4a', 'Chmp4b', 'Chmp4c',
    'Sdcbp', 'Pdcd6ip', 'Vta1', 'Stam', 'Hgs', 'Mvb12a', 'Mvb12b', 'Vps28',
    'Vps37a', 'Vps37b', 'Vps37c', 'Vps37d', 'Chmp1a', 'Chmp1b', 'Chmp2a',
    'Chmp2b', 'Chmp3', 'Chmp5', 'Chmp6', 'Chmp7', 'Ist1'
]

RISC_GENES_HUMAN = [
    'AGO1', 'AGO2', 'AGO3', 'AGO4', 'HSPA4', 'HSP90AA1', 'DICER1', 
    'HSP90AB1', 'HSPA8', 'HOPX', 'DNAJA2', 'PTGES3'
]

RISC_GENES_MOUSE = [
    'Ago1', 'Ago2', 'Ago3', 'Ago4', 'Hspa4', 'Hsp90aa1', 'Dicer1',
    'Hsp90ab1', 'Hspa8', 'Hopx', 'Dnaja2', 'Ptges3'
]

RITAC_GENES_HUMAN = [
    'AGO2', 'DHX9', 'PAF1', 'LEO1', 'CTR9', 'CDC73', 'RTF1', 'SKIC8',
    'SUPT4H1', 'SUPT5H', 'SUPT16H', 'SSRP1', 'TCEA1', 'TCEA2', 'TCEA3'
]

RITAC_GENES_MOUSE = [
    'Ago2', 'Dhx9', 'Paf1', 'Leo1', 'Ctr9', 'Cdc73', 'Rtf1', 'Skic8',
    'Supt4h1', 'Supt5h', 'Supt16h', 'Ssrp1', 'Tcea1', 'Tcea2', 'Tcea3'
]


def compute_module_score(X: np.ndarray, gene_indices: List[int], 
                         n_bins: int = 25, n_ctrl: int = 100) -> np.ndarray:
    n_cells, n_genes = X.shape
    
    if len(gene_indices) == 0:
        return np.zeros(n_cells)
    
    gene_means = np.mean(X, axis=0)
    bins = pd.cut(gene_means, bins=n_bins, labels=False)
    bins = np.nan_to_num(bins, nan=0).astype(int)
    
    target_expr = X[:, gene_indices].mean(axis=1)
    
    ctrl_genes = []
    for idx in gene_indices:
        bin_idx = bins[idx]
        same_bin = np.where(bins == bin_idx)[0]
        same_bin = same_bin[same_bin != idx]
        
        if len(same_bin) > 0:
            n_select = min(n_ctrl, len(same_bin))
            selected = np.random.choice(same_bin, n_select, replace=False)
            ctrl_genes.extend(selected)
    
    if len(ctrl_genes) > 0:
        ctrl_genes = list(set(ctrl_genes))
        ctrl_expr = X[:, ctrl_genes].mean(axis=1)
    else:
        ctrl_expr = np.zeros(n_cells)
    
    module_score = target_expr - ctrl_expr
    return module_score


def min_max_scale(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    if x_max - x_min == 0:
        return np.ones_like(x) * 0.5
    return (x - x_min) / (x_max - x_min)


def compute_proxy_scores(adata: ad.AnnData, mirna_targets: Dict[str, List[str]], 
                         method: str = 'zscore', min_targets: int = 3) -> Tuple[np.ndarray, List[str]]:
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_cells = X.shape[0]
    
    valid_mirnas = []
    proxy_scores = []
    
    if method == 'zscore':
        X_zscore = zscore(X, axis=0, nan_policy='omit')
        X_zscore = np.nan_to_num(X_zscore, nan=0.0)
        
        for mirna, targets in mirna_targets.items():
            target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(target_idx) < min_targets:
                continue
            target_zscore = X_zscore[:, target_idx]
            proxy = -np.mean(target_zscore, axis=1)
            valid_mirnas.append(mirna)
            proxy_scores.append(proxy)
    else:
        for mirna, targets in mirna_targets.items():
            target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(target_idx) < min_targets:
                continue
            proxy = X[:, target_idx].mean(axis=1)
            valid_mirnas.append(mirna)
            proxy_scores.append(proxy)
    
    if not valid_mirnas:
        return np.zeros((n_cells, 0)), []
    
    proxy_matrix = np.column_stack(proxy_scores)
    return proxy_matrix, valid_mirnas


class MiRTalkRestored:
    
    def __init__(self, 
                 top_k_edges: int = 10000,
                 min_targets: int = 3,
                 proxy_method: str = 'zscore',
                 n_permutations: int = 0,
                 p_threshold: float = 0.05,
                 species: str = 'human',
                 regulation: str = 'negative'):
        self.top_k_edges = top_k_edges
        self.min_targets = min_targets
        self.proxy_method = proxy_method
        self.n_permutations = n_permutations
        self.p_threshold = p_threshold
        self.species = species.lower()
        self.regulation = regulation
        
        self.mirna_edge_df = None
        self.cellpair_df = None
        self.celltype_df = None
        
        self._setup_gene_sets()
    
    def _setup_gene_sets(self):
        if self.species in ['human', 'homo sapiens']:
            self.evbs_genes = EVBS_GENES_HUMAN
            self.risc_genes = RISC_GENES_HUMAN
            self.ritac_genes = RITAC_GENES_HUMAN
        elif self.species in ['mouse', 'mus musculus']:
            self.evbs_genes = EVBS_GENES_MOUSE
            self.risc_genes = RISC_GENES_MOUSE
            self.ritac_genes = RITAC_GENES_MOUSE
        else:
            self.evbs_genes = [g.capitalize() for g in EVBS_GENES_MOUSE]
            self.risc_genes = [g.capitalize() for g in RISC_GENES_MOUSE]
            self.ritac_genes = [g.capitalize() for g in RITAC_GENES_MOUSE]
    
    def _find_gene_indices(self, gene_to_idx: Dict[str, int], gene_list: List[str]) -> List[int]:
        indices = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
        if len(indices) == 0:
            gene_lower = {g.lower(): i for g, i in gene_to_idx.items()}
            indices = [gene_lower[g.lower()] for g in gene_list if g.lower() in gene_lower]
        return indices
    
    def _compute_evbs_score(self, X: np.ndarray, gene_to_idx: Dict[str, int]) -> np.ndarray:
        evbs_idx = self._find_gene_indices(gene_to_idx, self.evbs_genes)
        if len(evbs_idx) == 0:
            return np.ones(X.shape[0]) * 0.5
        score = compute_module_score(X, evbs_idx)
        return min_max_scale(score)
    
    def _compute_risc_score(self, X: np.ndarray, gene_to_idx: Dict[str, int]) -> np.ndarray:
        risc_idx = self._find_gene_indices(gene_to_idx, self.risc_genes)
        if len(risc_idx) == 0:
            return np.ones(X.shape[0]) * 0.5
        score = compute_module_score(X, risc_idx)
        return min_max_scale(score)
    
    def _compute_ritac_score(self, X: np.ndarray, gene_to_idx: Dict[str, int]) -> np.ndarray:
        ritac_idx = self._find_gene_indices(gene_to_idx, self.ritac_genes)
        if len(ritac_idx) == 0:
            return np.ones(X.shape[0]) * 0.5
        score = compute_module_score(X, ritac_idx)
        return min_max_scale(score)
    
    def _compute_receiver_score_cell(self, X: np.ndarray, cell_idx: int, 
                                      target_idx: int, risc_val: float,
                                      n_genes: int) -> float:
        cell_expr = X[cell_idx, :]
        descending_order = np.argsort(-cell_expr)
        ranks = np.zeros(n_genes)
        ranks[descending_order] = np.arange(n_genes)
        N_j = ranks[target_idx] / (n_genes - 1) if n_genes > 1 else 0.5
        return N_j * risc_val
    
    def _compute_receiver_score_cell_positive(self, X: np.ndarray, cell_idx: int,
                                               target_idx: int, ritac_val: float,
                                               n_genes: int) -> float:
        cell_expr = X[cell_idx, :]
        ascending_order = np.argsort(cell_expr)
        ranks = np.zeros(n_genes)
        ranks[ascending_order] = np.arange(n_genes)
        P_j = ranks[target_idx] / (n_genes - 1) if n_genes > 1 else 0.5
        return P_j * ritac_val
    
    def _compute_sender_score_celltype(self, proxy_mat: np.ndarray, evbs_score: np.ndarray,
                                        cell_indices: np.ndarray, mirna_idx: int) -> float:
        mirna_expr = proxy_mat[cell_indices, mirna_idx]
        evbs = evbs_score[cell_indices]
        phi = mirna_expr * evbs
        mean_phi = np.mean(phi)
        high_cells_mask = phi > mean_phi
        if np.sum(high_cells_mask) == 0:
            return np.mean(phi)
        return np.mean(phi[high_cells_mask])
    
    def _compute_receiver_score_celltype(self, X: np.ndarray, risc_score: np.ndarray,
                                          cell_indices: np.ndarray, target_idx: int) -> float:
        n_cells = len(cell_indices)
        n_genes = X.shape[1]
        tau_values = np.zeros(n_cells)
        for i, cell_idx in enumerate(cell_indices):
            tau_values[i] = self._compute_receiver_score_cell(
                X, cell_idx, target_idx, risc_score[cell_idx], n_genes
            )
        mean_tau = np.mean(tau_values)
        high_cells_mask = tau_values > mean_tau
        if np.sum(high_cells_mask) == 0:
            return np.mean(tau_values)
        return np.mean(tau_values[high_cells_mask])
    
    def _permutation_test(self, real_score: float, X: np.ndarray,
                          proxy_mat: np.ndarray, evbs_score: np.ndarray,
                          risc_score: np.ndarray,
                          sender_indices: np.ndarray, receiver_indices: np.ndarray,
                          mirna_idx: int, target_idx: int,
                          cell_types: np.ndarray) -> float:
        n_cells = len(cell_types)
        permuted_scores = np.zeros(self.n_permutations)
        sender_type = cell_types[sender_indices[0]]
        receiver_type = cell_types[receiver_indices[0]]
        
        for p in range(self.n_permutations):
            perm_labels = np.random.permutation(cell_types)
            perm_sender_idx = np.where(perm_labels == sender_type)[0]
            perm_receiver_idx = np.where(perm_labels == receiver_type)[0]
            
            if len(perm_sender_idx) == 0 or len(perm_receiver_idx) == 0:
                permuted_scores[p] = 0
                continue
            
            perm_sender_score = self._compute_sender_score_celltype(
                proxy_mat, evbs_score, perm_sender_idx, mirna_idx
            )
            perm_receiver_score = self._compute_receiver_score_celltype(
                X, risc_score, perm_receiver_idx, target_idx
            )
            permuted_scores[p] = perm_sender_score * perm_receiver_score
        
        p_value = np.mean(permuted_scores >= real_score)
        return p_value
    
    def fit_predict(self, adata: ad.AnnData, 
                    mirna_targets: Dict[str, List[str]],
                    biogenesis_genes: List[str] = None,
                    risc_genes: List[str] = None,
                    sorting_genes: List[str] = None,
                    gt_labels: pd.DataFrame = None) -> pd.DataFrame:
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        n_cells, n_genes = X.shape
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        cell_types = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None
        
        proxy_mat, mirna_names = compute_proxy_scores(
            adata, mirna_targets, self.proxy_method, self.min_targets
        )
        
        if proxy_mat.shape[1] == 0:
            mirna_names = []
            proxy_cols = []
            for mirna, targets in (mirna_targets or {}).items():
                if not targets:
                    continue
                target_idx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
                if len(target_idx) < self.min_targets:
                    continue
                mirna_names.append(mirna)
                proxy_cols.append(X[:, target_idx].mean(axis=1))
            
            if len(mirna_names) == 0:
                return pd.DataFrame()
            proxy_mat = np.stack(proxy_cols, axis=1)
        
        proxy_norm = np.apply_along_axis(min_max_scale, 0, proxy_mat)
        evbs_score = self._compute_evbs_score(X, gene_to_idx)
        
        if biogenesis_genes:
            biog_idx = self._find_gene_indices(gene_to_idx, biogenesis_genes)
            if len(biog_idx) > 0:
                evbs_score = min_max_scale(compute_module_score(X, biog_idx))
        
        if self.regulation == 'negative':
            receiver_module_score = self._compute_risc_score(X, gene_to_idx)
            if risc_genes:
                risc_idx = self._find_gene_indices(gene_to_idx, risc_genes)
                if len(risc_idx) > 0:
                    receiver_module_score = min_max_scale(compute_module_score(X, risc_idx))
        else:
            receiver_module_score = self._compute_ritac_score(X, gene_to_idx)
        
        use_gt_pairs = gt_labels is not None and len(gt_labels) > 0
        
        if use_gt_pairs:
            senders = gt_labels['sender_cell'].values.astype(int)
            receivers = gt_labels['receiver_cell'].values.astype(int)
            
            scores = []
            edges = []
            
            for idx, (s, r) in enumerate(zip(senders, receivers)):
                pair_mirna_scores = []
                
                for j, mirna in enumerate(mirna_names):
                    sender_score = proxy_norm[s, j] * evbs_score[s]
                    targets = mirna_targets.get(mirna, [])
                    target_indices = [gene_to_idx[t] for t in targets if t in gene_to_idx]
                    
                    if len(target_indices) < self.min_targets:
                        continue
                    
                    target_scores = []
                    for target_idx in target_indices[:10]:
                        if self.regulation == 'negative':
                            recv_score = self._compute_receiver_score_cell(
                                X, r, target_idx, receiver_module_score[r], n_genes
                            )
                        else:
                            recv_score = self._compute_receiver_score_cell_positive(
                                X, r, target_idx, receiver_module_score[r], n_genes
                            )
                        target_scores.append(recv_score)
                    
                    if target_scores:
                        receiver_score = np.mean(target_scores)
                        comm_score = sender_score * receiver_score
                        pair_mirna_scores.append(comm_score)
                        
                        edge = {
                            'sender_cell': s,
                            'receiver_cell': r,
                            'mirna': mirna,
                            'score': comm_score,
                            'sender_score': sender_score,
                            'receiver_score': receiver_score,
                        }
                        if cell_types is not None:
                            edge['sender_type'] = cell_types[s]
                            edge['receiver_type'] = cell_types[r]
                        edges.append(edge)
                
                if pair_mirna_scores:
                    scores.append(np.max(pair_mirna_scores))
                else:
                    scores.append(0.0)
            
            self.cellpair_df = pd.DataFrame({
                'sender_cell': senders,
                'receiver_cell': receivers,
                'score': scores
            })
            if cell_types is not None:
                self.cellpair_df['sender_type'] = cell_types[senders]
                self.cellpair_df['receiver_type'] = cell_types[receivers]
            
            self.mirna_edge_df = pd.DataFrame(edges) if edges else pd.DataFrame()
            
        else:
            if cell_types is None:
                raise ValueError("cell_type column required when gt_labels not provided")
            
            unique_cell_types = np.unique(cell_types)
            edges = []
            
            for sender_type in unique_cell_types:
                sender_idx = np.where(cell_types == sender_type)[0]
                
                for receiver_type in unique_cell_types:
                    receiver_idx = np.where(cell_types == receiver_type)[0]
                    
                    for j, mirna in enumerate(mirna_names):
                        targets = mirna_targets.get(mirna, [])
                        target_indices = [gene_to_idx[t] for t in targets if t in gene_to_idx]
                        
                        if len(target_indices) < self.min_targets:
                            continue
                        
                        for target_idx in target_indices:
                            target_gene = gene_names[target_idx]
                            
                            sender_score = self._compute_sender_score_celltype(
                                proxy_norm, evbs_score, sender_idx, j
                            )
                            receiver_score = self._compute_receiver_score_celltype(
                                X, receiver_module_score, receiver_idx, target_idx
                            )
                            comm_score = sender_score * receiver_score
                            
                            if self.n_permutations > 0:
                                p_value = self._permutation_test(
                                    comm_score, X, proxy_norm, evbs_score,
                                    receiver_module_score,
                                    sender_idx, receiver_idx,
                                    j, target_idx, cell_types
                                )
                            else:
                                p_value = np.nan
                            
                            if comm_score > 0.01 or not np.isnan(p_value):
                                edges.append({
                                    'sender_type': sender_type,
                                    'receiver_type': receiver_type,
                                    'mirna': mirna,
                                    'target': target_gene,
                                    'sender_score': sender_score,
                                    'receiver_score': receiver_score,
                                    'score': comm_score,
                                    'p_value': p_value
                                })
            
            if not edges:
                return pd.DataFrame()
            
            self.mirna_edge_df = pd.DataFrame(edges)
            
            if len(self.mirna_edge_df) > self.top_k_edges:
                self.mirna_edge_df = self.mirna_edge_df.nlargest(self.top_k_edges, 'score')
            
            self.mirna_edge_df = self.mirna_edge_df.reset_index(drop=True)
        
        return self.mirna_edge_df
    
    def get_cellpair_df(self, agg: str = 'max') -> pd.DataFrame:
        if self.cellpair_df is not None and len(self.cellpair_df) > 0:
            return self.cellpair_df
        
        if self.mirna_edge_df is None or len(self.mirna_edge_df) == 0:
            return pd.DataFrame()
        
        if 'sender_cell' in self.mirna_edge_df.columns:
            group_cols = ['sender_cell', 'receiver_cell']
            if 'sender_type' in self.mirna_edge_df.columns:
                group_cols.extend(['sender_type', 'receiver_type'])
        else:
            group_cols = ['sender_type', 'receiver_type']
        
        if agg == 'sum':
            df = self.mirna_edge_df.groupby(group_cols)['score'].sum().reset_index()
        elif agg == 'max':
            df = self.mirna_edge_df.groupby(group_cols)['score'].max().reset_index()
        else:
            df = self.mirna_edge_df.groupby(group_cols)['score'].mean().reset_index()
        
        self.cellpair_df = df.sort_values('score', ascending=False).reset_index(drop=True)
        return self.cellpair_df
    
    def get_celltype_pair_df(self) -> pd.DataFrame:
        if self.cellpair_df is None:
            self.get_cellpair_df()
        
        if self.cellpair_df is None or 'sender_type' not in self.cellpair_df.columns:
            if self.mirna_edge_df is not None and 'sender_type' in self.mirna_edge_df.columns:
                return self.mirna_edge_df.groupby(
                    ['sender_type', 'receiver_type']
                ).agg({'score': 'sum'}).reset_index()
            return pd.DataFrame()
        
        return self.cellpair_df.groupby(
            ['sender_type', 'receiver_type']
        ).agg({'score': 'sum'}).reset_index()


def run_mirtalk_restored(adata: ad.AnnData, out_dir: str, top_k: int = 10000,
                         gt_labels: pd.DataFrame = None,
                         n_permutations: int = 0,
                         species: str = 'human') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    
    mirna_targets = adata.uns.get('mirna_targets', {})
    biog = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    sort = adata.uns.get('sorting_genes', [])
    
    model = MiRTalkRestored(
        top_k_edges=top_k,
        n_permutations=n_permutations,
        species=species
    )
    
    mirna_df = model.fit_predict(adata, mirna_targets, biog, risc, sort, gt_labels)
    cellpair_df = model.get_cellpair_df()
    celltype_df = model.get_celltype_pair_df()
    
    if mirna_df is not None and len(mirna_df) > 0:
        mirna_df.to_csv(os.path.join(out_dir, 'mirtalk_restored_mirna.csv'), index=False)
    if cellpair_df is not None and len(cellpair_df) > 0:
        cellpair_df.to_csv(os.path.join(out_dir, 'mirtalk_restored_cellpair.csv'), index=False)
    if celltype_df is not None and len(celltype_df) > 0:
        celltype_df.to_csv(os.path.join(out_dir, 'mirtalk_restored_celltype.csv'), index=False)
    
    return mirna_df, cellpair_df, celltype_df


MiRTalkLite = MiRTalkRestored


def run_mirtalk_lite(adata: ad.AnnData, out_dir: str, top_k: int = 10000,
                     gt_labels: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_mirtalk_restored(
        adata, out_dir, top_k, gt_labels,
        n_permutations=0,
        species='human'
    )


if __name__ == '__main__':
    print("miRTalk-Restored module loaded successfully")
    print(f"EVBS genes (human): {len(EVBS_GENES_HUMAN)}")
    print(f"RISC genes (human): {len(RISC_GENES_HUMAN)}")
    print(f"RITAC genes (human): {len(RITAC_GENES_HUMAN)}")
