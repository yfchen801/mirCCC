import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import zscore
from typing import Dict, List, Optional, Tuple
import warnings

from .proxy_baselines import compute_sender_receiver_proxy


def _trimean(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    q1 = np.percentile(arr, 25)
    q2 = np.percentile(arr, 50)
    q3 = np.percentile(arr, 75)
    return (q1 + 2.0 * q2 + q3) / 4.0


def _hill(x: float, K_h: float = 0.5, n: float = 1.0) -> float:
    if x <= 0:
        return 0.0
    xn = x ** n
    return xn / (K_h ** n + xn)


class CellChatStyleBaseline:

    def __init__(self, K_h: float = 0.5, n_permutations: int = 100):
        self.K_h = K_h
        self.n_permutations = n_permutations
        self.cellpair_df = None
        self.comm_prob = None

    def fit_predict(self, adata: ad.AnnData,
                    mirna_targets: Dict[str, List[str]],
                    biogenesis_genes: List[str],
                    risc_genes: List[str],
                    sorting_genes: List[str] = None,
                    gt_labels: pd.DataFrame = None) -> pd.DataFrame:

        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        n_cells = X.shape[0]
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        cell_types = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None

        if cell_types is None:
            warnings.warn("CellChat-Style requires cell_type annotation. Falling back to single group.")
            cell_types = np.array(['all'] * n_cells)

        groups = np.unique(cell_types)
        group_indices = {g: np.where(cell_types == g)[0] for g in groups}

        emit_scores, _ = compute_sender_receiver_proxy(
            adata, biogenesis_genes, risc_genes, sorting_genes
        )
        e_min, e_max = emit_scores.min(), emit_scores.max()
        if e_max > e_min:
            emit_norm = (emit_scores - e_min) / (e_max - e_min)
        else:
            emit_norm = np.ones(n_cells) * 0.5

        valid_mirnas = []
        repression_cols = []
        for mirna, targets in mirna_targets.items():
            tidx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(tidx) < 3:
                continue
            target_expr = X[:, tidx].mean(axis=1)
            te_min, te_max = target_expr.min(), target_expr.max()
            if te_max > te_min:
                repress = 1.0 - (target_expr - te_min) / (te_max - te_min)
            else:
                repress = np.zeros(n_cells)
            valid_mirnas.append(mirna)
            repression_cols.append(repress)

        n_mirna = len(valid_mirnas)
        print(f"    [CellChat-Style] {n_mirna} valid miRNAs, {len(groups)} cell groups")

        if n_mirna == 0:
            self.cellpair_df = pd.DataFrame(columns=['sender_cell', 'receiver_cell', 'score'])
            return self.cellpair_df

        repression_mat = np.column_stack(repression_cols)

        n_groups = len(groups)
        group_to_idx = {g: i for i, g in enumerate(groups)}

        L_group = np.zeros((n_groups, n_mirna))
        R_group = np.zeros((n_groups, n_mirna))
        for gi, g in enumerate(groups):
            cidx = group_indices[g]
            for mi in range(n_mirna):
                L_group[gi, mi] = _trimean(emit_norm[cidx])
                R_group[gi, mi] = _trimean(repression_mat[cidx, mi])

        comm_prob = np.zeros((n_groups, n_groups))
        for mi in range(n_mirna):
            for si in range(n_groups):
                for ri in range(n_groups):
                    lr_product = L_group[si, mi] * R_group[ri, mi]
                    comm_prob[si, ri] += _hill(lr_product, self.K_h)

        self.comm_prob = pd.DataFrame(comm_prob, index=groups, columns=groups)

        if self.n_permutations > 0:
            null_dist = np.zeros((self.n_permutations, n_groups, n_groups))
            for perm in range(self.n_permutations):
                perm_labels = np.random.permutation(cell_types)
                perm_indices = {g: np.where(perm_labels == g)[0] for g in groups}
                for mi in range(n_mirna):
                    for si in range(n_groups):
                        for ri in range(n_groups):
                            L_perm = _trimean(emit_norm[perm_indices[groups[si]]])
                            R_perm = _trimean(repression_mat[perm_indices[groups[ri]], mi])
                            null_dist[perm, si, ri] += _hill(L_perm * R_perm, self.K_h)

            p_values = np.mean(null_dist >= comm_prob[np.newaxis, :, :], axis=0)
            comm_prob[p_values >= 0.05] = 0.0
            n_sig = np.sum(p_values < 0.05)
            print(f"    [CellChat-Style] {n_sig}/{n_groups*n_groups} significant group pairs (p<0.05)")

        if gt_labels is not None and len(gt_labels) > 0:
            senders = gt_labels['sender_cell'].values.astype(int)
            receivers = gt_labels['receiver_cell'].values.astype(int)
        else:
            n_sample = min(10000, n_cells * n_cells)
            senders = np.random.choice(n_cells, size=n_sample)
            receivers = np.random.choice(n_cells, size=n_sample)
            mask = senders != receivers
            senders, receivers = senders[mask], receivers[mask]

        scores = []
        for s, r in zip(senders, receivers):
            sg = group_to_idx[cell_types[s]]
            rg = group_to_idx[cell_types[r]]
            scores.append(comm_prob[sg, rg])

        records = {
            'sender_cell': senders,
            'receiver_cell': receivers,
            'score': scores
        }
        if cell_types is not None:
            records['sender_type'] = cell_types[senders]
            records['receiver_type'] = cell_types[receivers]

        self.cellpair_df = pd.DataFrame(records)
        self.cellpair_df = self.cellpair_df.sort_values('score', ascending=False).reset_index(drop=True)

        scores_arr = np.array(scores)
        print(f"    [CellChat-Style] score range: [{scores_arr.min():.4f}, {scores_arr.max():.4f}], "
              f"n_pairs: {len(self.cellpair_df)}")

        return self.cellpair_df

    def get_celltype_pair_df(self) -> pd.DataFrame:
        if self.cellpair_df is None or 'sender_type' not in self.cellpair_df.columns:
            return pd.DataFrame()
        return self.cellpair_df.groupby(
            ['sender_type', 'receiver_type']
        ).agg({'score': 'sum'}).reset_index()


class NicheNetStyleBaseline:

    def __init__(self, damping: float = 0.5, n_walk_steps: int = 3):
        self.damping = damping
        self.n_walk_steps = n_walk_steps
        self.cellpair_df = None
        self.ligand_activity = None

    def _compute_regulatory_potential(self, mirna_targets: Dict[str, List[str]],
                                       gene_names: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        n_genes = len(gene_names)

        valid_mirnas = []
        mirna_target_idx = {}
        for m, targets in mirna_targets.items():
            tidx = [gene_to_idx[t] for t in targets if t in gene_to_idx]
            if len(tidx) >= 3:
                valid_mirnas.append(m)
                mirna_target_idx[m] = tidx

        n_mirna = len(valid_mirnas)
        if n_mirna == 0:
            return np.zeros((0, n_genes)), [], gene_names

        reg_potential = np.zeros((n_mirna, n_genes))
        for mi, m in enumerate(valid_mirnas):
            for gi in mirna_target_idx[m]:
                reg_potential[mi, gi] = 1.0

        for step in range(1, self.n_walk_steps):
            decay = self.damping ** step
            new_potential = np.zeros_like(reg_potential)
            for mi, m in enumerate(valid_mirnas):
                direct = mirna_target_idx[m]
                for m2 in valid_mirnas:
                    if m2 == m:
                        continue
                    shared = set(mirna_target_idx[m]) & set(mirna_target_idx[m2])
                    if len(shared) > 0:
                        unique_m2 = set(mirna_target_idx[m2]) - set(direct)
                        propagation_weight = len(shared) / len(mirna_target_idx[m2])
                        for gi in unique_m2:
                            new_potential[mi, gi] = max(
                                new_potential[mi, gi],
                                decay * propagation_weight
                            )
            reg_potential += new_potential

        for mi in range(n_mirna):
            rmax = reg_potential[mi].max()
            if rmax > 0:
                reg_potential[mi] /= rmax

        return reg_potential, valid_mirnas, gene_names

    def fit_predict(self, adata: ad.AnnData,
                    mirna_targets: Dict[str, List[str]],
                    biogenesis_genes: List[str],
                    risc_genes: List[str],
                    sorting_genes: List[str] = None,
                    gt_labels: pd.DataFrame = None) -> pd.DataFrame:

        X = adata.X.toarray() if sparse.issparse(adata.X) else np.array(adata.X)
        n_cells, n_genes = X.shape
        gene_names = list(adata.var_names)
        cell_types = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None

        reg_potential, valid_mirnas, _ = self._compute_regulatory_potential(
            mirna_targets, gene_names
        )
        n_mirna = len(valid_mirnas)
        print(f"    [NicheNet-Style] {n_mirna} valid miRNAs, regulatory potential computed")

        if n_mirna == 0:
            self.cellpair_df = pd.DataFrame(columns=['sender_cell', 'receiver_cell', 'score'])
            return self.cellpair_df

        emit_scores, _ = compute_sender_receiver_proxy(
            adata, biogenesis_genes, risc_genes, sorting_genes
        )
        e_min, e_max = emit_scores.min(), emit_scores.max()
        if e_max > e_min:
            emit_norm = (emit_scores - e_min) / (e_max - e_min)
        else:
            emit_norm = np.ones(n_cells) * 0.5

        X_ranked = np.zeros_like(X)
        for i in range(n_cells):
            X_ranked[i] = np.argsort(np.argsort(X[i])).astype(float) / n_genes

        X_repress = 1.0 - X_ranked

        ligand_activity = X_repress @ reg_potential.T

        for mi in range(n_mirna):
            col = ligand_activity[:, mi]
            cmin, cmax = col.min(), col.max()
            if cmax > cmin:
                ligand_activity[:, mi] = (col - cmin) / (cmax - cmin)
            else:
                ligand_activity[:, mi] = 0.0

        self.ligand_activity = ligand_activity

        receiver_total_activity = ligand_activity.sum(axis=1)

        ra_min, ra_max = receiver_total_activity.min(), receiver_total_activity.max()
        if ra_max > ra_min:
            recv_norm = (receiver_total_activity - ra_min) / (ra_max - ra_min)
        else:
            recv_norm = np.ones(n_cells) * 0.5

        if gt_labels is not None and len(gt_labels) > 0:
            senders = gt_labels['sender_cell'].values.astype(int)
            receivers = gt_labels['receiver_cell'].values.astype(int)
        else:
            n_sample = min(10000, n_cells * n_cells)
            senders = np.random.choice(n_cells, size=n_sample)
            receivers = np.random.choice(n_cells, size=n_sample)
            mask = senders != receivers
            senders, receivers = senders[mask], receivers[mask]

        scores = emit_norm[senders] * recv_norm[receivers]

        records = {
            'sender_cell': senders,
            'receiver_cell': receivers,
            'score': scores
        }
        if cell_types is not None:
            records['sender_type'] = cell_types[senders]
            records['receiver_type'] = cell_types[receivers]

        self.cellpair_df = pd.DataFrame(records)
        self.cellpair_df = self.cellpair_df.sort_values('score', ascending=False).reset_index(drop=True)

        print(f"    [NicheNet-Style] score range: [{scores.min():.4f}, {scores.max():.4f}], "
              f"n_pairs: {len(self.cellpair_df)}")

        return self.cellpair_df

    def get_celltype_pair_df(self) -> pd.DataFrame:
        if self.cellpair_df is None or 'sender_type' not in self.cellpair_df.columns:
            return pd.DataFrame()
        return self.cellpair_df.groupby(
            ['sender_type', 'receiver_type']
        ).agg({'score': 'sum'}).reset_index()


def run_cellchat_style(adata: ad.AnnData, out_dir: str,
                       K_h: float = 0.5, n_permutations: int = 100,
                       gt_labels: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)

    mirna_targets = adata.uns.get('mirna_targets', {})
    biog = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    sort = adata.uns.get('sorting_genes', [])

    print(f"    [DEBUG CellChat] mirna_targets: {len(mirna_targets)}, "
          f"biogenesis: {len(biog)}, risc: {len(risc)}")

    model = CellChatStyleBaseline(K_h=K_h, n_permutations=n_permutations)
    cellpair_df = model.fit_predict(adata, mirna_targets, biog, risc, sort, gt_labels)
    celltype_df = model.get_celltype_pair_df()

    if len(cellpair_df) > 0:
        cellpair_df.to_csv(os.path.join(out_dir, 'cellchat_style_cellpair.csv'), index=False)
    if len(celltype_df) > 0:
        celltype_df.to_csv(os.path.join(out_dir, 'cellchat_style_celltype.csv'), index=False)

    return cellpair_df, celltype_df


def run_nichenet_style(adata: ad.AnnData, out_dir: str,
                       damping: float = 0.5, n_walk_steps: int = 3,
                       gt_labels: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)

    mirna_targets = adata.uns.get('mirna_targets', {})
    biog = adata.uns.get('biogenesis_genes', [])
    risc = adata.uns.get('risc_genes', [])
    sort = adata.uns.get('sorting_genes', [])

    print(f"    [DEBUG NicheNet] mirna_targets: {len(mirna_targets)}, "
          f"biogenesis: {len(biog)}, risc: {len(risc)}")

    model = NicheNetStyleBaseline(damping=damping, n_walk_steps=n_walk_steps)
    cellpair_df = model.fit_predict(adata, mirna_targets, biog, risc, sort, gt_labels)
    celltype_df = model.get_celltype_pair_df()

    if len(cellpair_df) > 0:
        cellpair_df.to_csv(os.path.join(out_dir, 'nichenet_style_cellpair.csv'), index=False)
    if len(celltype_df) > 0:
        celltype_df.to_csv(os.path.join(out_dir, 'nichenet_style_celltype.csv'), index=False)

    return cellpair_df, celltype_df
