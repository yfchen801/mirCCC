
import os
import sys
import time
import re
import json
import pickle
import gzip
import warnings
from pathlib import Path
from datetime import datetime
from argparse import Namespace
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import nbinom, pearsonr, spearmanr, wilcoxon
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm, Normalize
import seaborn as sns

import scanpy as sc
import anndata as ad
import h5py

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import TransformerConv, DeepGraphInfomax
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader, DataLoader, ClusterData
    from torch_geometric.utils import to_undirected, k_hop_subgraph
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch/PyG not available. Training steps will be skipped.")

try:
    import umap
    UMAP = umap.UMAP
except Exception:
    try:
        from umap.umap_ import UMAP
    except:
        UMAP = None

warnings.filterwarnings('ignore')


class Config:
    
    _MIRNA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mirna")
    MIR2TAR_PATH = os.path.join(_MIRNA_DIR, "mir2tar.csv")
    BIOGENESIS_PATH = os.path.join(_MIRNA_DIR, "biogenesis.csv")
    RISC_PATH = os.path.join(_MIRNA_DIR, "risc.csv")
    SORTING_PATH = os.path.join(_MIRNA_DIR, "sorting.csv")
    GENEINFO_PATH = os.path.join(_MIRNA_DIR, "geneinfo.csv")
    SPECIES = "Human"
    
    OUTPUT_DIR = "./mirage_output"
    
    SYNTH_N_CELLS = 3000
    SYNTH_N_GENES = 4000
    SYNTH_N_CELL_TYPES = 8
    SYNTH_SPARSITY = 0.45
    SYNTH_NOISE_LEVEL = 0.15
    SYNTH_BATCH_EFFECT = 0.0
    SYNTH_RANDOM_SEED = 42
    
    MIN_CELLS_EXPR = 5
    MIN_TARGETS_PER_MIRNA = 10
    COV_MIN = 0
    GAMMA_COV = 2.5
    
    TOP_K_MIRNA_PER_CELL = 0
    COEXPR_THRESHOLD = 0.005
    MIN_TARGETS = 5
    TOP_S_SENDERS = 150
    TOP_R_RECEIVERS = 150
    TOP_L_PER_SENDER = 10
    
    NUM_EPOCHS = 500
    PRINT_EVERY = 50
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    MIR_EMB_DIM = 16
    DEVICE = "cuda:0" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    NUM_PARTS = 50
    OVERLAP_HOPS = 1
    RETAIN_PERCENT = 1
    
    SAVE_FIGURES = True
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'


def setup_output_dirs(config: Config):
    dirs = [
        config.OUTPUT_DIR,
        f"{config.OUTPUT_DIR}/synthetic_data",
        f"{config.OUTPUT_DIR}/figures",
        f"{config.OUTPUT_DIR}/model",
        f"{config.OUTPUT_DIR}/results",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


class MiRNATargetDatabase:
    
    def __init__(
        self, 
        mir2tar_path: str,
        species: str = "Human",
        min_targets: int = 5,
        use_strong_evidence_only: bool = False
    ):
        self.mir2tar_path = mir2tar_path
        self.species = species
        self.min_targets = min_targets
        self.use_strong_evidence_only = use_strong_evidence_only
        
        self.raw_df = self._load_data()
        self.interactions = self._build_interactions()
        self.mirnas = list(self.interactions.index)
        self.target_genes = list(self.interactions.columns)
        
    def _load_data(self) -> pd.DataFrame:
        print(f"Loading mir2tar from: {self.mir2tar_path}")
        
        df = pd.read_csv(self.mir2tar_path)
        print(f"  Raw records: {len(df)}")
        
        df = df[df['species'] == self.species].copy()
        print(f"  After species filter ({self.species}): {len(df)}")
        
        df = df[df['regulation'] == 'Negative'].copy()
        print(f"  After regulation filter (Negative): {len(df)}")
        
        if self.use_strong_evidence_only:
            df = df[~df['support'].str.contains('Weak', na=False)].copy()
            print(f"  After strong evidence filter: {len(df)}")
        
        return df
    
    def _build_interactions(self) -> pd.DataFrame:
        df = self.raw_df.copy()
        df['mirna'] = df['miRNA_mature'].fillna(df['miRNA'])
        
        mirna_target_counts = df.groupby('mirna')['target_gene'].nunique()
        valid_mirnas = mirna_target_counts[mirna_target_counts >= self.min_targets].index.tolist()
        df = df[df['mirna'].isin(valid_mirnas)].copy()
        
        print(f"  Valid miRNAs (>={self.min_targets} targets): {len(valid_mirnas)}")
        
        def get_confidence(support):
            if pd.isna(support):
                return 0.5
            return 0.5 if 'weak' in str(support).lower() else 1.0
        
        df['confidence'] = df['support'].apply(get_confidence)
        
        interaction_df = df.groupby(['mirna', 'target_gene'])['confidence'].max().reset_index()
        interaction_matrix = interaction_df.pivot(
            index='mirna', columns='target_gene', values='confidence'
        ).fillna(0)
        
        print(f"  Interaction matrix shape: {interaction_matrix.shape}")
        print(f"  Total interactions: {(interaction_matrix > 0).sum().sum()}")
        
        return interaction_matrix
    
    def get_targets(self, mirna: str, confidence_threshold: float = 0.5) -> List[str]:
        if mirna not in self.interactions.index:
            return []
        targets = self.interactions.loc[mirna]
        return targets[targets >= confidence_threshold].index.tolist()
    
    def get_top_mirnas(self, n: int = 30) -> List[str]:
        n_targets = (self.interactions > 0).sum(axis=1).sort_values(ascending=False)
        return n_targets.head(n).index.tolist()
    
    def summary(self):
        n_high = (self.interactions == 1.0).sum().sum()
        n_medium = (self.interactions == 0.5).sum().sum()
        
        print(f"\nmiRNA-Target Database Summary:")
        print(f"  miRNAs: {len(self.mirnas)}")
        print(f"  Target genes: {len(self.target_genes)}")
        print(f"  High confidence: {n_high}")
        print(f"  Medium confidence: {n_medium}")


class CommunicationPattern:
    
    def __init__(self, mirna_db: MiRNATargetDatabase):
        self.mirna_db = mirna_db
        self.cell_types = [
            'T_cell', 'B_cell', 'NK_cell', 'Monocyte', 
            'Macrophage', 'Dendritic_cell', 'Neutrophil', 'Tumor_cell'
        ]
        self.available_mirnas = set(mirna_db.mirnas)
        self.communication_patterns = self._define_patterns()
    
    def _get_valid_mirnas(self, mirna_list: List[str]) -> List[str]:
        valid = []
        for m in mirna_list:
            if m in self.available_mirnas:
                valid.append(m)
            else:
                base = m.replace('-3p', '').replace('-5p', '')
                matches = [am for am in self.available_mirnas if base in am]
                if matches:
                    valid.append(matches[0])
        return valid
    
    def _define_patterns(self) -> List[Dict]:
        MAX_T = 25
        MIRNAS_PER_PATTERN = 3
        rng = np.random.default_rng(42)

        tgt_cnt = (self.mirna_db.interactions >= 0.5).sum(axis=1).astype(int)
        pool = tgt_cnt[(tgt_cnt >= 10) & (tgt_cnt <= MAX_T)].index.tolist()
        if len(pool) < MIRNAS_PER_PATTERN * 10:
            print(f"[WARN] small-target miRNA pool is small: {len(pool)}")

        patterns_def = [
            ('Tumor_cell', 'T_cell', 0.8),
            ('Tumor_cell', 'Macrophage', 0.75),
            ('Macrophage', 'Tumor_cell', 0.65),
            ('T_cell', 'B_cell', 0.6),
            ('Monocyte', 'Macrophage', 0.7),
            ('NK_cell', 'Tumor_cell', 0.55),
            ('Dendritic_cell', 'T_cell', 0.7),
            ('Neutrophil', 'Tumor_cell', 0.5),
            ('Tumor_cell', 'NK_cell', 0.6),
            ('B_cell', 'Dendritic_cell', 0.45),
        ]

        rng.shuffle(pool)
        ptr = 0
        validated = []
        for sender, receiver, strength in patterns_def:
            if ptr + MIRNAS_PER_PATTERN > len(pool):
                rng.shuffle(pool)
                ptr = 0
            mirnas = pool[ptr:ptr + MIRNAS_PER_PATTERN]
            ptr += MIRNAS_PER_PATTERN
            validated.append({
                "sender": sender,
                "receiver": receiver,
                "mirnas": mirnas,
                "effect_strength": strength,
            })
        return validated

    
    def get_communication_matrix(self) -> pd.DataFrame:
        matrix = pd.DataFrame(0.0, index=self.cell_types, columns=self.cell_types)
        for p in self.communication_patterns:
            matrix.loc[p['sender'], p['receiver']] = p['effect_strength']
        return matrix


class SyntheticDataGenerator:
    def __init__(
        self,
        mirna_db: MiRNATargetDatabase,
        n_cells: int = 5000,
        n_genes: int = 5000,
        n_cell_types: int = 8,
        sparsity: float = 0.85,
        noise_level: float = 0.3,
        batch_effect_strength: float = 0.0,
        n_batches: int = 3,
        random_seed: int = 42,
        biogenesis_path: str = None,
        risc_path: str = None,
        sorting_path: str = None,
        species: str = "Human",
        mirna_coverage: float = 1.0,
        sr_ratio: str = "5:5",
        prior_boost: float = 3.0,
        
        repression_strength: float = 0.2,
        
        protect_dropout_scale: float = 0.25,
        transfer_prob_base: float = 0.85
    ):
        self.mirna_db = mirna_db
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.batch_effect_strength = batch_effect_strength
        self.n_batches = n_batches
        self.random_seed = random_seed

        self.biogenesis_path = biogenesis_path
        self.risc_path = risc_path
        self.sorting_path = sorting_path
        self.species = species

        self.mirna_coverage = float(np.clip(mirna_coverage, 0.0, 1.0))
        self.sr_ratio = sr_ratio
        self.prior_boost = float(prior_boost)
        
        self.repression_strength = float(repression_strength)
        
        self.protect_dropout_scale = float(protect_dropout_scale)
        self.transfer_prob_base = float(transfer_prob_base)

        np.random.seed(random_seed)

        self.comm_patterns = CommunicationPattern(mirna_db)
        self.cell_types = self.comm_patterns.cell_types[:n_cell_types]

        self.biogenesis_genes = self._read_gene_set(self.biogenesis_path)
        self.risc_genes = self._read_gene_set(self.risc_path)
        self.sorting_genes = self._read_gene_set(self.sorting_path)

        self.genes = self._build_gene_list()

        self.signal_gene_set = set(self.mirna_db.target_genes) | \
                               set(self.biogenesis_genes) | set(self.risc_genes) | set(self.sorting_genes)

        self.ground_truth = {
            'communication_pairs': [],
            'mirna_activity': {},
            'cell_assignments': None,
        }

    def _read_gene_set(self, path: str) -> List[str]:
        if path is None or (isinstance(path, str) and path.strip() == ""):
            return []
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)

        if "species" in df.columns and self.species is not None:
            df = df[df["species"].astype(str) == str(self.species)].copy()

        for col in ["gene", "symbol", "gene_name", "Gene", "SYMBOL"]:
            if col in df.columns:
                gcol = col
                break
        else:
            gcol = df.columns[0]

        genes = df[gcol].dropna().astype(str).str.strip().tolist()

        if str(self.species).lower().startswith("human"):
            genes = [g.upper() for g in genes if g]

        return sorted(set(genes))

    def _build_gene_list(self) -> List[str]:
        def _read_gene_list(path: str) -> List[str]:
            if (path is None) or (not os.path.exists(path)):
                return []
            df = pd.read_csv(path)
            for c in ["gene", "symbol", "Gene", "GENE", "gene_name"]:
                if c in df.columns:
                    return df[c].astype(str).tolist()
            return df.iloc[:, 0].astype(str).tolist()

        used_mirnas = sorted({m for p in self.comm_patterns.communication_patterns for m in p["mirnas"]})
        targets = []
        for m in used_mirnas:
            targets.extend(self.mirna_db.get_targets(m))
        targets = list(dict.fromkeys([t for t in targets if isinstance(t, str) and t != ""]))

        bio = _read_gene_list(Config.BIOGENESIS_PATH)
        risc = _read_gene_list(Config.RISC_PATH)
        sortg = _read_gene_list(Config.SORTING_PATH)
        modules = list(dict.fromkeys([*bio, *risc, *sortg]))

        genes = list(dict.fromkeys([*modules, *targets]))

        n_markers_total = min(200, max(0, self.n_genes - len(genes)))
        for i in range(n_markers_total):
            genes.append(f"CTMARK_{i:04d}")

        if len(genes) < self.n_genes:
            remaining = [g for g in self.mirna_db.target_genes if g not in set(genes)]
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(remaining)
            genes.extend(remaining)

        if len(genes) < self.n_genes:
            prefixes = ['GENE', 'LOC', 'FAM', 'KIAA', 'ORF', 'LINC', 'AC', 'AL']
            j = 0
            while len(genes) < self.n_genes:
                genes.append(f"{prefixes[j % len(prefixes)]}{j+1:05d}")
                j += 1

        return genes[:self.n_genes]

    
    def _generate_base_expression(self) -> np.ndarray:
        print("  [1/6] Generating base expression...")
        
        n_per_type = self.n_cells // len(self.cell_types)
        expression = np.zeros((self.n_cells, self.n_genes))
        cell_type_labels = []
        
        idx = 0
        for ct_idx, ct in enumerate(self.cell_types):
            n_ct = n_per_type if ct_idx < len(self.cell_types) - 1 else self.n_cells - idx
            type_factor = np.random.gamma(2, 0.5, self.n_genes)
            
            for _ in range(n_ct):
                cell_means = 2.5 * type_factor * np.random.gamma(2, 0.5)
                r, p = 2.0, 2.0 / (2.0 + cell_means + 1e-10)
                expression[idx] = nbinom.rvs(r, np.clip(p, 0.01, 0.99))
                cell_type_labels.append(ct)
                idx += 1
        
        self.cell_type_labels = cell_type_labels
        self.ground_truth['cell_assignments'] = cell_type_labels.copy()
        return expression
    
    def _add_cell_type_markers(self, expression: np.ndarray) -> np.ndarray:
        print("  [2/6] Adding cell type markers...")

        marker_idx = [i for i, g in enumerate(self.genes) if str(g).startswith("CTMARK_")]
        if len(marker_idx) == 0:
            return expression

        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(marker_idx)

        markers_per_type = max(5, min(20, len(marker_idx) // max(1, len(self.cell_types))))
        ptr = 0

        for ct in self.cell_types:
            ct_cells = [i for i, c in enumerate(self.cell_type_labels) if c == ct]
            if not ct_cells:
                continue
            sel = marker_idx[ptr:ptr + markers_per_type]
            ptr += markers_per_type
            if len(sel) == 0:
                break

            boost = rng.uniform(3, 8, size=len(sel))
            for ci in ct_cells:
                expression[ci, sel] *= boost

        return expression

    
    def _parse_ratio(self, ratio: str) -> Tuple[float, float]:
        try:
            a, b = str(ratio).split(":")
            a = float(a); b = float(b)
            s = a / (a + b + 1e-12)
            r = b / (a + b + 1e-12)
            return float(np.clip(s, 0.05, 0.95)), float(np.clip(r, 0.05, 0.95))
        except:
            return 0.5, 0.5

    def _embed_mirna_communication(self, expression: np.ndarray) -> np.ndarray:
        print("  [3/6] Embedding miRNA communication signals (Global Target Boost -> Specific Repression)...")

        gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        biog_idx = [gene_to_idx[g] for g in self.biogenesis_genes if g in gene_to_idx]
        risc_idx = [gene_to_idx[g] for g in self.risc_genes if g in gene_to_idx]
        sort_idx = [gene_to_idx[g] for g in self.sorting_genes if g in gene_to_idx]

        sender_frac, recv_frac = self._parse_ratio(self.sr_ratio)
        
        active_sender_indices = set()
        active_receiver_indices = set()

        for pattern in self.comm_patterns.communication_patterns:
            sender, receiver = pattern['sender'], pattern['receiver']
            if sender not in self.cell_types or receiver not in self.cell_types:
                continue

            mirnas_all = list(pattern['mirnas'])
            strength = float(pattern['effect_strength'])

            if len(mirnas_all) == 0: continue
            k = max(1, int(np.ceil(len(mirnas_all) * max(self.mirna_coverage, 1e-6))))
            mirnas = list(np.random.choice(mirnas_all, size=min(k, len(mirnas_all)), replace=False))

            sender_idx_all = [i for i, c in enumerate(self.cell_type_labels) if c == sender]
            receiver_idx_all = [i for i, c in enumerate(self.cell_type_labels) if c == receiver]
            
            if not sender_idx_all or not receiver_idx_all: continue

            s_active_n = max(1, int(len(sender_idx_all) * sender_frac))
            r_active_n = max(1, int(len(receiver_idx_all) * recv_frac))
            
            sender_active = list(np.random.choice(sender_idx_all, size=s_active_n, replace=False))
            receiver_active = list(np.random.choice(receiver_idx_all, size=r_active_n, replace=False))
            
            active_sender_indices.update(sender_active)
            active_receiver_indices.update(receiver_active)

            real_boost = self.prior_boost * 2.0  
            if biog_idx:
                expression[np.ix_(sender_active, biog_idx)] = (expression[np.ix_(sender_active, biog_idx)] + 0.5) * real_boost
            if sort_idx:
                expression[np.ix_(sender_active, sort_idx)] = (expression[np.ix_(sender_active, sort_idx)] + 0.5) * real_boost
            if risc_idx:
                expression[np.ix_(receiver_active, risc_idx)] = (expression[np.ix_(receiver_active, risc_idx)] + 0.5) * real_boost

            for m in mirnas:
                targets = [t for t in self.mirna_db.get_targets(m) if t in gene_to_idx]
                if not targets: continue
                ti = [gene_to_idx[t] for t in targets]
                
                pre_boost_val = 5.0 
                target_cells = sender_idx_all + receiver_idx_all
                expression[np.ix_(target_cells, ti)] = np.maximum(
                    expression[np.ix_(target_cells, ti)], 
                    np.random.poisson(pre_boost_val, size=(len(target_cells), len(ti)))
                )

                for recv in receiver_active:
                    expression[recv, ti] *= self.repression_strength

                cis_repress_factor = 0.25 
                for snd in sender_active:
                    expression[snd, ti] *= cis_repress_factor

                if m not in self.ground_truth['mirna_activity']:
                    self.ground_truth['mirna_activity'][m] = {
                        'sender_cells': [], 'receiver_cells': [], 'n_targets': len(targets)
                    }
                self.ground_truth['mirna_activity'][m]['sender_cells'].extend(sender_active)
                self.ground_truth['mirna_activity'][m]['receiver_cells'].extend(receiver_active)
                
            self.ground_truth['communication_pairs'].append({
                'sender': sender, 'receiver': receiver, 'mirnas': mirnas, 'strength': strength,
                'sender_n_cells': len(sender_active), 'receiver_n_cells': len(receiver_active)
            })

        print(f"       Injecting Decoys (Fake Senders/Receivers)...")
        
        all_indices = set(range(self.n_cells))
        used_indices = active_sender_indices | active_receiver_indices
        unused_indices = list(all_indices - used_indices)
        
        if len(unused_indices) < 100:
            unused_indices = list(np.random.choice(self.n_cells, size=min(self.n_cells, 500), replace=False))
            
        np.random.shuffle(unused_indices)
        split_pt = len(unused_indices) // 2
        fake_senders = unused_indices[:split_pt]
        fake_receivers = unused_indices[split_pt:]

        decoy_scale = 1.0 
        
        if len(fake_senders) > 0:
            fake_boost = self.prior_boost * decoy_scale
            if biog_idx: 
                expression[np.ix_(fake_senders, biog_idx)] *= fake_boost
            if sort_idx: 
                expression[np.ix_(fake_senders, sort_idx)] *= fake_boost

        if len(fake_receivers) > 0:
            fake_boost = self.prior_boost * decoy_scale
            if risc_idx: 
                expression[np.ix_(fake_receivers, risc_idx)] *= fake_boost
            
            non_target_genes = list(set(range(self.n_genes)) - self.signal_gene_set)
            if len(non_target_genes) > 50:
                random_repression_idx = np.random.choice(non_target_genes, size=20, replace=False)
                expression[np.ix_(fake_receivers, random_repression_idx)] *= 0.2

        print(f"       Embedded patterns & decoys complete.")
        return expression
    
    def _add_dropout(self, expression: np.ndarray) -> np.ndarray:
        print(f"  [4/6] Adding dropout (sparsity={self.sparsity})...")

        log_expr = np.log1p(expression)
        normalized = log_expr / (np.max(log_expr) + 1e-10)
        dropout_prob = self.sparsity * (1 - normalized ** 0.4)

        protect = np.array([g in self.signal_gene_set for g in self.genes], dtype=bool)
        if protect.any():
            dropout_prob[:, protect] *= self.protect_dropout_scale
        
        expression[np.random.random(expression.shape) < dropout_prob] = 0
        return expression

    
    def _add_noise(self, expression: np.ndarray) -> np.ndarray:
        print(f"  [5/6] Adding noise (level={self.noise_level})...")
        
        expression *= np.random.lognormal(0, self.noise_level * 0.8, expression.shape)
        expression += np.abs(np.random.normal(0, self.noise_level * 0.3, expression.shape))
        return expression
    
    def _add_batch_effect(self, expression: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        print(f"  [6/6] Adding batch effects (strength={self.batch_effect_strength})...")
        
        batch_labels = []
        per_batch = self.n_cells // self.n_batches
        
        for b in range(self.n_batches):
            start = b * per_batch
            end = self.n_cells if b == self.n_batches - 1 else start + per_batch
            
            if self.batch_effect_strength > 0:
                expression[start:end] *= np.random.lognormal(0, self.batch_effect_strength, self.n_genes)
            
            batch_labels.extend([f'batch_{b}'] * (end - start))
        
        return expression, batch_labels
    
    def generate(self) -> ad.AnnData:
        print("=" * 60)
        print("Generating Synthetic Data")
        print("=" * 60)
        
        expression = self._generate_base_expression()
        expression = self._add_cell_type_markers(expression)
        expression = self._embed_mirna_communication(expression)
        expression = self._add_noise(expression)
        expression = self._add_dropout(expression)
        expression, batch_labels = self._add_batch_effect(expression)
        
        expression = np.maximum(expression, 0).astype(np.float32)
        
        adata = ad.AnnData(
            X=sp.csr_matrix(expression),
            obs=pd.DataFrame({
                'cell_type': self.cell_type_labels,
                'batch': batch_labels,
            }),
            var=pd.DataFrame({
                'gene_name': self.genes,
                'is_target': [g in self.mirna_db.target_genes for g in self.genes]
            }, index=self.genes)
        )
        
        adata.uns['n_comm_patterns'] = len(self.ground_truth['communication_pairs'])
        adata.uns['n_mirnas_with_activity'] = len(self.ground_truth['mirna_activity'])
        adata.uns['generation_params'] = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'sparsity': self.sparsity,
            'noise_level': self.noise_level,
            'batch_effect_strength': self.batch_effect_strength,
            'random_seed': self.random_seed
        }
        
        sparsity = 1 - np.count_nonzero(adata.X.toarray()) / (adata.n_obs * adata.n_vars)
        
        print("\n" + "=" * 60)
        print("Done!")
        print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"  Patterns: {len(self.ground_truth['communication_pairs'])}")
        print("=" * 60)
        
        return adata
    
    def get_ground_truth(self) -> Dict:
        return self.ground_truth
    
    def generate_ground_truth_labels(self) -> pd.DataFrame:
        labels = []
        
        for p in self.ground_truth['communication_pairs']:
            sender_cells = [i for i, c in enumerate(self.cell_type_labels) if c == p['sender']]
            receiver_cells = [i for i, c in enumerate(self.cell_type_labels) if c == p['receiver']]
            
            n_pairs = min(200, len(sender_cells) * len(receiver_cells) // 10)
            for _ in range(n_pairs):
                labels.append({
                    'sender_cell': np.random.choice(sender_cells),
                    'receiver_cell': np.random.choice(receiver_cells),
                    'sender_type': p['sender'],
                    'receiver_type': p['receiver'],
                    'communication': 1,
                    'strength': p['strength'],
                    'mirnas': ','.join(p['mirnas'])
                })
        
        positive_pairs = {(p['sender'], p['receiver']) for p in self.ground_truth['communication_pairs']}
        negative_pairs = [(c1, c2) for c1 in self.cell_types for c2 in self.cell_types 
                          if (c1, c2) not in positive_pairs]
        
        n_neg = len([l for l in labels if l['communication'] == 1])
        for _ in range(n_neg):
            if not negative_pairs:
                break
            c1, c2 = negative_pairs[np.random.randint(len(negative_pairs))]
            s_cells = [i for i, c in enumerate(self.cell_type_labels) if c == c1]
            r_cells = [i for i, c in enumerate(self.cell_type_labels) if c == c2]
            if s_cells and r_cells:
                labels.append({
                    'sender_cell': np.random.choice(s_cells),
                    'receiver_cell': np.random.choice(r_cells),
                    'sender_type': c1,
                    'receiver_type': c2,
                    'communication': 0,
                    'strength': 0,
                    'mirnas': ''
                })
        
        return pd.DataFrame(labels)


import unicodedata as ud

_SPECIES_ALIASES = {
    "human": {"human", "Human"},
    "mouse": {"mouse", "Mouse"},
    "rat":   {"rat", "Rat"},
}

def canonical_species(s: str) -> str:
    s0 = str(s).strip().lower()
    s1 = " ".join(s0.replace("_", " ").replace("-", " ").replace(":", " ").split())
    for key, aliases in _SPECIES_ALIASES.items():
        if s0 in aliases or s1 in aliases or s0 == key:
            return key
    raise ValueError(f"Unsupported species: {s} (use human/mouse/rat)")


def symbol_case_by_species(sym: str, species: str) -> str:
    if not sym:
        return ""
    s = str(sym).strip()
    spc = canonical_species(species)
    if spc == "human":
        return s.upper()
    return (s[:1].upper() + "".join(ch.lower() for ch in s[1:]))


_STOP = {"", "-", ".", "Ã¢â‚¬â€", "Ã¢â‚¬â€œ", "Ã¢Ë†â€™", "Ã¢â‚¬â€™", "Ã¢â‚¬â€¢", "NA", "N/A", "NULL", "None", "?", "*"}

_DASH_MAP = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
})


def _clean_token(raw: str, species: str) -> str:
    if pd.isna(raw):
        return ""
    s = ud.normalize("NFKC", str(raw)).translate(_DASH_MAP)
    s = " ".join(s.strip().split())
    return symbol_case_by_species(s, species)


def _synonym_valid(tok: str) -> bool:
    if tok in _STOP:
        return False
    return sum(ch.isalnum() for ch in tok) >= 2


def _root(s: str) -> str:
    return re.sub(r"[-._]", "", s)


def build_alias_map_from_geneinfo(geneinfo_csv: str, species: str):
    sp_key = canonical_species(species)
    df = pd.read_csv(geneinfo_csv)

    need = {"symbol", "synonyms", "species"}
    if not need.issubset(df.columns):
        raise KeyError(f"geneinfo.csv must contain columns: {need}")

    def _is_species(x):
        try:
            return canonical_species(x) == sp_key
        except:
            return False
    df = df[df["species"].apply(_is_species)].copy()

    df["symbol_norm"] = df["symbol"].map(lambda x: _clean_token(x, sp_key))
    df["synonym_norm"] = df["synonyms"].map(lambda x: _clean_token(x, sp_key))
    df = df[df["synonym_norm"].apply(_synonym_valid)].copy()

    grouped = df.groupby("synonym_norm", sort=False)["symbol_norm"].agg(
        lambda s: sorted(set(s))
    ).reset_index()

    alias2sym = {}
    n_unique = 0
    n_conflict = 0

    for syn, syms in grouped.itertuples(index=False):
        if len(syms) == 1:
            alias2sym[syn] = syms[0]
            n_unique += 1
        else:
            syn_r = _root(syn)
            exact = [sym for sym in syms if _root(sym) == syn_r]
            if len(exact) == 1:
                alias2sym[syn] = exact[0]
                n_unique += 1
            else:
                n_conflict += 1
                continue

    for sym in df["symbol_norm"].unique():
        alias2sym.setdefault(sym, sym)

    stats = {
        "n_rows": len(df),
        "n_alias_unique": n_unique,
        "n_alias_conflict_skipped": n_conflict,
        "n_symbols": df["symbol_norm"].nunique(),
        "species": sp_key,
    }
    print(f"  [alias-map] species={sp_key} | rows={stats['n_rows']} | unique={n_unique} | "
          f"conflict_skipped={n_conflict} | symbols={stats['n_symbols']}")
    return alias2sym, stats


def unify_and_merge_matrix(W, gene_names, alias2sym: dict, species: str, agg: str = "sum"):
    if agg not in ("sum", "max"):
        raise ValueError("agg must be 'sum' or 'max'")
    sp_key = canonical_species(species)

    mapped = []
    for g in list(gene_names):
        g_std = symbol_case_by_species(g, sp_key)
        mapped.append(alias2sym.get(g_std, g_std))
    mapped = np.asarray(mapped, dtype=object)

    uniq_syms, new_col = np.unique(mapped, return_inverse=True)
    n_cells, n_new = (W.shape[0], len(uniq_syms))

    if sp.issparse(W):
        W = W.tocoo(copy=False)
        out = sp.coo_matrix((W.data, (W.row, new_col[W.col])), shape=(n_cells, n_new))
        W_new = out.tocsr()
    else:
        if agg == "sum":
            W_new = np.zeros((n_cells, n_new), dtype=W.dtype)
            for old_j, new_j in enumerate(new_col):
                W_new[:, new_j] += W[:, old_j]
        else:
            W_new = np.full((n_cells, n_new), -np.inf, dtype=W.dtype)
            for old_j, new_j in enumerate(new_col):
                W_new[:, new_j] = np.maximum(W_new[:, new_j], W[:, old_j])
            W_new[~np.isfinite(W_new)] = 0

    report = {
        "n_genes_before": len(gene_names),
        "n_genes_after": len(uniq_syms),
        "n_collapsed": len(gene_names) - len(uniq_syms)
    }
    print(f"  [unify] {sp_key}: genes {report['n_genes_before']} -> {report['n_genes_after']} "
          f"(merged {report['n_collapsed']})")
    return W_new, uniq_syms.tolist(), report


def unify_anndata_varnames(adata, geneinfo_csv: str, species: str, agg: str = "sum", save_path: str = None):
    if adata is None or not hasattr(adata, "X"):
        raise ValueError("Invalid AnnData object")
    
    print("\n  Unifying gene names...")
    alias2sym, stats = build_alias_map_from_geneinfo(geneinfo_csv, species)
    X_new, genes_new, rep = unify_and_merge_matrix(
        adata.X, adata.var_names.astype(str).tolist(),
        alias2sym, species, agg=agg
    )
    
    adata_new = ad.AnnData(
        X_new,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(genes_new, name="gene"))
    )
    
    if hasattr(adata, 'obsm'):
        adata_new.obsm = adata.obsm.copy()
    if hasattr(adata, 'uns'):
        adata_new.uns = adata.uns.copy()
    
    adata_new.uns["gene_alias_map"] = {
        "synonyms_to_symbol": alias2sym,
        "stats": stats,
        "unify_report": rep
    }

    if save_path is not None:
        adata_new.write_h5ad(save_path)
        print(f"  [save] Unified AnnData saved to: {save_path}")

    return adata_new, {"alias_stats": stats, "unify_report": rep}

def support_to_score(s):
    if pd.isna(s): 
        return 0.0
    s = str(s).strip().lower()
    sc = 0.0
    if "primary-confidence: 1" in s: 
        sc = max(sc, 1.0)
    if "functional mti" in s and "weak" not in s: 
        sc = max(sc, 1.0)
    if "functional mti (weak)" in s: 
        sc = max(sc, 0.1)
    if "secondary-confidence: 1" in s: 
        sc = max(sc, 0.1)
    if "non-functional mti (weak)" in s: 
        sc = max(sc, 0.01)
    if "non-functional mti" in s: 
        sc = max(sc, 0.5)
    return float(sc)


def compute_mirna_coverage_weight(
    mirna_list,
    gene_list,
    W,
    mir2tar_csv,
    min_detect_frac: float = 0.05,
    alpha: float = 1.0,
) -> np.ndarray:
    mirna_list = list(map(str, mirna_list))
    gene_list = np.array(gene_list, dtype=str)

    df = pd.read_csv(mir2tar_csv)

    if "species" in df.columns:
        df = df[df["species"].astype(str).str.strip().str.lower() == "human"].copy()

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        raise KeyError(f"CSV missing column: {cols}")

    col_mi = pick(["miRNA", "mirna", "Mirna", "miR", "mi_rna"])
    col_tg = pick(["target_gene", "target", "gene", "Gene"])
    col_sup = "support" if "support" in df.columns else pick(
        ["support", "Support.Type", "support_type", "evidence", "Evidence"]
    )

    df["miRNA"] = df[col_mi].astype(str)
    df["target_gene"] = df[col_tg].astype(str)
    df["support_raw"] = df[col_sup]
    df["s_score"] = df["support_raw"].map(support_to_score).astype(float)
    df = df[df["s_score"] > 0.0].copy()
    
    if df.empty:
        return np.ones(len(mirna_list), dtype=np.float32)

    df_all = df.copy()
    gene_set = set(gene_list.tolist())
    df_obs = df[df["target_gene"].isin(gene_set)].copy()

    df_all_uniq = df_all.groupby(["miRNA", "target_gene"], as_index=False)["s_score"].max()
    S_all = df_all_uniq.groupby("miRNA")["s_score"].sum()

    if sp.issparse(W):
        nz = np.asarray((W > 0).sum(axis=0)).ravel().astype(float)
    else:
        nz = (W > 0).sum(axis=0).astype(float)

    n_cells = W.shape[0]
    detect_frac = nz / max(n_cells, 1)
    gene_index = pd.Index(gene_list)

    df_obs["gene_idx"] = gene_index.get_indexer(df_obs["target_gene"].values)
    df_obs = df_obs[df_obs["gene_idx"] >= 0].copy()
    
    if df_obs.empty:
        return np.ones(len(mirna_list), dtype=np.float32)

    df_obs["detect_frac"] = detect_frac[df_obs["gene_idx"].values]
    df_obs_det = df_obs[df_obs["detect_frac"] >= min_detect_frac].copy()
    
    if df_obs_det.empty:
        return np.zeros(len(mirna_list), dtype=np.float32)

    df_obs_uniq = df_obs_det.groupby(["miRNA", "target_gene"], as_index=False)["s_score"].max()
    S_obs = df_obs_uniq.groupby("miRNA")["s_score"].sum()

    q_cov = np.zeros(len(mirna_list), dtype=np.float32)
    for i, m in enumerate(mirna_list):
        if m not in S_all.index:
            q_cov[i] = 0.0
            continue
        denom = float(S_all[m])
        if denom <= 0:
            q_cov[i] = 0.0
            continue
        num = float(S_obs[m]) if m in S_obs.index else 0.0
        cov = num / denom
        cov = max(0.0, min(1.0, cov))
        q_cov[i] = cov ** alpha

    if q_cov.max() == 0:
        q_cov[:] = 1.0

    return q_cov


def reweight_A_signed_with_support(A_signed, mir2tar_csv, mirna_list, gene_list, row_norm="l1", use_idf=True):
    mir2idx = {m: i for i, m in enumerate(mirna_list)}
    gene2idx = {g: j for j, g in enumerate(gene_list)}
    M, G = len(mirna_list), len(gene_list)
    df = pd.read_csv(mir2tar_csv)

    def pick(opts):
        for c in opts:
            if c in df.columns: 
                return c
        raise KeyError(f"CSV missing column: {opts}")

    col_mi = pick(["miRNA", "mirna", "Mirna", "miR", "mi_rna"])
    col_tg = pick(["target_gene", "target", "gene", "Gene"])
    col_reg = pick(["regulation", "effect", "type"])
    col_sup = "support" if "support" in df.columns else pick(
        ["support", "Support.Type", "support_type", "evidence", "Evidence"]
    )

    if "species" in df.columns:
        df = df[df["species"].astype(str).str.strip().str.lower() == "human"].copy()

    df = df[df[col_mi].isin(mir2idx) & df[col_tg].isin(gene2idx)].copy()
    if df.empty: 
        return sp.csr_matrix((M, G))
    
    reg = df[col_reg].astype(str).str.lower()
    df["sign"] = np.where(reg.str.contains("neg"), 1.0, np.where(reg.str.contains("pos"), -1.0, 0.0))
    df["score"] = df[col_sup].map(support_to_score)
    grp = df.groupby([col_mi, col_tg], as_index=False).agg(score=("score", "max"), sign=("sign", "max"))
    grp["val"] = grp["score"] * grp["sign"]
    grp = grp[grp["val"] != 0.0]
    
    if grp.empty: 
        return sp.csr_matrix((M, G))
    
    grp["i"] = grp[col_mi].map(mir2idx).astype(int)
    grp["j"] = grp[col_tg].map(gene2idx).astype(int)
    A = sp.csr_matrix((grp["val"].values, (grp["i"].values, grp["j"].values)), shape=(M, G))
    
    if use_idf:
        nnz_per_gene = np.asarray((A != 0).sum(axis=0)).ravel().astype(float)
        nnz_per_gene = np.maximum(nnz_per_gene, 1.0)
        idf = np.log(M / nnz_per_gene + 1.0)
        A = A @ sp.diags(idf)
    
    if row_norm == "l1":
        row_sum = np.asarray(np.abs(A).sum(axis=1)).ravel()
        row_sum = np.maximum(row_sum, 1e-12)
        A = sp.diags(1.0 / row_sum) @ A
    elif row_norm == "l2":
        row_sq = np.asarray((A.power(2)).sum(axis=1)).ravel()
        row_norm_vals = np.sqrt(np.maximum(row_sq, 1e-12))
        A = sp.diags(1.0 / row_norm_vals) @ A
    
    return A


def compute_zscore_matrix(W, eps: float = 1e-12):
    if sp.issparse(W):
        mu = np.asarray(W.mean(axis=0)).ravel()
        sq = np.asarray(W.power(2).mean(axis=0)).ravel()
        var = np.maximum(sq - mu**2, eps)
        scale = 1.0 / np.sqrt(var)
        return W @ sp.diags(scale)
    else:
        X = np.asarray(W, dtype=np.float64, order="C")
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True)
        std = np.maximum(std, eps)
        return (X - mean) / std


def build_signal_matrix(X, M: int, sparse: bool = False, dtype=None):
    C, G = X.shape[0], X.shape[1]
    out_dtype = dtype or (X.dtype if hasattr(X, "dtype") else np.float32)

    if sparse:
        zero_m = sp.csr_matrix((M, C), dtype=out_dtype)
        gene_expr = X.T if sp.issparse(X) else sp.csr_matrix(X.T, dtype=out_dtype)
        S0 = sp.vstack([zero_m, gene_expr], format="csr", dtype=out_dtype)
    else:
        Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
        zero_m = np.zeros((M, C), dtype=out_dtype)
        gene_expr = Xd.T.astype(out_dtype, copy=False)
        S0 = np.vstack([zero_m, gene_expr])

    return S0


def build_propagation_adjacency(A_signed: sp.csr_matrix, normalize: bool = True) -> sp.csr_matrix:
    M, G = A_signed.shape
    zero_MM = sp.csr_matrix((M, M))
    zero_GG = sp.csr_matrix((G, G))

    top = sp.hstack([zero_MM, A_signed])
    bottom = sp.hstack([A_signed.T, zero_GG])
    A_full = sp.vstack([top, bottom]).tocsr()

    if normalize:
        deg = np.array(A_full.sum(axis=1)).flatten()
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt[np.isnan(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        A_full = D_inv_sqrt @ A_full @ D_inv_sqrt

    return A_full


def propagate_signed_onestep(A_signed: sp.csr_matrix, S0: np.ndarray) -> np.ndarray:
    deg_abs = np.array(np.abs(A_signed).sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        deg_inv = 1.0 / deg_abs
    deg_inv[np.isinf(deg_inv)] = 0.0
    D_inv = sp.diags(deg_inv)

    A_rw_signed = D_inv @ A_signed
    S1 = -(A_rw_signed @ S0)
    return S1


def compute_mirna_proxy(adata, mir2tar_csv, config: Config):
    print("\n" + "=" * 60)
    print("STEP 1: Computing miRNA Proxy")
    print("=" * 60)
    
    X = adata.X
    if sp.issparse(X):
        W_raw = X.toarray()
    else:
        W_raw = np.asarray(X)
    
    gene_list = adata.var_names.tolist()
    cell_names = adata.obs_names.tolist()
    
    df = pd.read_csv(mir2tar_csv)
    df = df[df["species"].astype(str).str.strip().str.lower() == "human"].copy()
    
    col_mi = next(c for c in ["miRNA", "mirna", "Mirna"] if c in df.columns)
    col_tg = next(c for c in ["target_gene", "target", "gene"] if c in df.columns)
    
    targets = set(df[col_tg].astype(str))
    nz_cnt = (W_raw > 0).sum(axis=0)
    expressed = set(np.array(gene_list)[nz_cnt >= config.MIN_CELLS_EXPR])
    keep = sorted(targets & expressed)
    
    if not keep:
        raise ValueError("No target genes found in expression data")
    
    idx = pd.Index(gene_list).get_indexer(keep)
    W = W_raw[:, idx]
    gene_list_filtered = keep
    
    df_sub = df[df[col_tg].astype(str).isin(set(gene_list_filtered))].copy()
    df_sub_uniq = df_sub[[col_mi, col_tg]].drop_duplicates()
    tgt_cnt = df_sub_uniq.groupby(col_mi)[col_tg].nunique()
    keep_mirs = tgt_cnt[tgt_cnt >= config.MIN_TARGETS_PER_MIRNA].index.astype(str).tolist()
    
    if not keep_mirs:
        raise ValueError(f"No miRNA has >= {config.MIN_TARGETS_PER_MIRNA} targets")
    
    mirna_list = sorted(keep_mirs)
    print(f"  Filtered genes: {len(gene_list_filtered)}")
    print(f"  Filtered miRNAs: {len(mirna_list)}")
    
    A_ph = sp.csr_matrix((len(mirna_list), len(gene_list_filtered)))
    A_signed = reweight_A_signed_with_support(
        A_ph, mir2tar_csv, mirna_list, gene_list_filtered,
        row_norm="l1", use_idf=True
    )
    
    W_z = compute_zscore_matrix(W)
    
    M = len(mirna_list)
    S0 = build_signal_matrix(W_z, M=M)
    A_propagation = build_propagation_adjacency(A_signed, normalize=True)
    S_diffused = propagate_signed_onestep(A_propagation, S0)
    
    q_cov = compute_mirna_coverage_weight(
        mirna_list=mirna_list,
        gene_list=gene_list_filtered,
        W=W_raw[:, idx],
        mir2tar_csv=mir2tar_csv,
        min_detect_frac=0.05,
        alpha=1.0
    )
    
    keep = (q_cov >= config.COV_MIN)
    print(f"  miRNAs with coverage >= {config.COV_MIN}: {keep.sum()} / {M}")

    adata.uns["miRNA_proxy_qcov"] = q_cov
    adata.uns["miRNA_proxy_keep"] = keep
    adata.uns["miRNA_proxy_cov_min"] = config.COV_MIN

    S_filt = S_diffused.copy()
    mask_keep = keep.astype(S_filt.dtype)

    S_filt[:M, :] *= mask_keep[:, None]
    S_filt[:M, :] *= (q_cov ** config.GAMMA_COV)[:, None].astype(S_filt.dtype)

    PR = pd.DataFrame(
        S_filt[:M, :].T,
        index=pd.Index(cell_names, dtype=str),
        columns=pd.Index(mirna_list, dtype=str)
    )

    if getattr(config, "PROXY_KEEP_ONLY", False):
        PR = PR.loc[:, keep]
        mirna_list = PR.columns.astype(str).tolist()
        q_cov = q_cov[keep]
        adata.uns["miRNA_proxy_qcov"] = q_cov
        adata.uns["miRNA_proxy_keep"] = np.ones(len(mirna_list), dtype=bool)

    X_proxy = PR.to_numpy(dtype=np.float32)
    X_proxy = np.nan_to_num(X_proxy, nan=0.0, posinf=0.0, neginf=0.0)

    adata.obsm["X_miRNA_proxy"] = X_proxy
    adata.uns["miRNA_proxy_names"] = PR.columns.astype(str).tolist()
    
    print(f"  Proxy matrix shape: {X_proxy.shape}")
    print(f"  Stored in adata.obsm['X_miRNA_proxy']")
    
    return adata, PR


def z01(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mx - mn) or (mx - mn) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def mean_zscore_score(X_sub):
    X_sub = np.asarray(X_sub, dtype=np.float32)
    mu = X_sub.mean(axis=0, keepdims=True)
    sigma = X_sub.std(axis=0, ddof=1, keepdims=True)
    sigma[sigma < 1e-6] = 1e-6
    Z = (X_sub - mu) / sigma
    return Z.mean(axis=1).astype(np.float32), mu.ravel(), sigma.ravel()


def compute_sender_receiver_scores(adata, config: Config):
    print("\n" + "=" * 60)
    print("STEP 2: Computing Sender/Receiver Scores")
    print("=" * 60)
    
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    
    varnames = adata.var_names.to_numpy()
    var_up = {g.upper(): i for i, g in enumerate(varnames)}
    
    def map_to_var(genelist_upper):
        idx, hit, miss = [], [], []
        for g in genelist_upper:
            i = var_up.get(g, None)
            if i is None:
                miss.append(g)
            else:
                idx.append(i)
                hit.append(varnames[i])
        return np.array(idx, dtype=int), hit, miss
    
    def read_human_genes(filepath, name):
        if not os.path.exists(filepath):
            print(f"  [WARNING] {name} file not found: {filepath}")
            return np.array([])
        df = pd.read_csv(filepath)
        if "species" in df.columns:
            sub = df[df["species"].astype(str).str.lower() == "human"]
        else:
            sub = df
        if "gene" not in sub.columns:
            print(f"  [WARNING] {name} missing 'gene' column")
            return np.array([])
        return sub["gene"].astype(str).str.upper().unique()
    
    biog_genes = read_human_genes(config.BIOGENESIS_PATH, "biogenesis")
    uptk_genes = read_human_genes(config.RISC_PATH, "risc")
    sort_genes = read_human_genes(config.SORTING_PATH, "sorting")
    
    idx_biog, biog_hit, _ = map_to_var(biog_genes)
    idx_uptk, uptk_hit, _ = map_to_var(uptk_genes)
    idx_sort, sort_hit, _ = map_to_var(sort_genes)
    
    print(f"  Biogenesis genes: {len(biog_hit)} hit")
    print(f"  RISC genes: {len(uptk_hit)} hit")
    print(f"  Sorting genes: {len(sort_hit)} hit")
    
    def compute_score(idx_arr, name):
        if len(idx_arr) == 0:
            print(f"  [WARNING] No {name} genes found, using zeros")
            return np.zeros(adata.n_obs, dtype=np.float32)
        X_sub = adata[:, idx_arr].X
        X_sub = X_sub.toarray() if sp.issparse(X_sub) else np.asarray(X_sub)
        score, _, _ = mean_zscore_score(X_sub.astype(np.float32))
        return score
    
    release_score = compute_score(idx_biog, "biogenesis")
    risc_score = compute_score(idx_uptk, "risc")
    sorting_score = compute_score(idx_sort, "sorting")
    
    adata.obs["release_score"] = release_score
    adata.obs["risc_score"] = risc_score
    adata.obs["sorting_score"] = sorting_score
    
    adata.obs["release_score_01"] = z01(release_score)
    adata.obs["risc_score_01"] = z01(risc_score)
    adata.obs["sorting_score_01"] = z01(sorting_score)
    
    emit_prior_01 = (adata.obs["release_score_01"].to_numpy(np.float32) * 
                     adata.obs["sorting_score_01"].to_numpy(np.float32))
    adata.obs["emit_prior_01"] = z01(emit_prior_01)
    
    adata.obs["sender_score"] = release_score
    adata.obs["receiver_score"] = risc_score
    if "release_score" in adata.obs and "biogenesis_score" not in adata.obs:
        adata.obs["biogenesis_score"] = adata.obs["release_score"].values

    if "release_score_01" in adata.obs and "biogenesis_score_01" not in adata.obs:
        adata.obs["biogenesis_score_01"] = adata.obs["release_score_01"].values

    print(f"  Computed scores stored in adata.obs")
    
    return adata


def _norm_mir_name(s):
    s = str(s).strip().lower().replace("hsa-", "")
    s = s.replace("*", "")
    return re.sub(r"-(3p|5p)$", "", s)


def assign_confidence_weight(row):
    W_HIGH = 1.0
    W_MED = 0.5
    W_LOW = 0.1

    support = str(row.get('support', '')).strip()

    try:
        primary_conf = int(row.get('primary-confidence', -1))
    except (ValueError, TypeError):
        primary_conf = -1

    try:
        primary = int(row.get('primary', -1))
    except (ValueError, TypeError):
        primary = -1

    try:
        secondary = int(row.get('secondary', -1))
    except (ValueError, TypeError):
        secondary = -1

    if (support == "Functional MTI") or (primary_conf == 1):
        return W_HIGH
    if (support in ["Functional MTI (Weak)", "Non-Functional MTI"]) or (primary == 0 and secondary == 1):
        return W_MED
    if (support == "Non-Functional MTI (Weak)") or (primary == 0 and secondary == 0):
        return W_LOW
    return W_LOW


def build_communication_graph(adata, mir2tar_csv, config: Config):
    print("\n" + "=" * 60)
    print("STEP 3: Building Communication Graph")
    print("=" * 60)
    
    t0 = time.time()
    
    if "X_miRNA_proxy" not in adata.obsm:
        raise KeyError("X_miRNA_proxy not found. Run compute_mirna_proxy first.")
    
    proxy_arr = np.asarray(adata.obsm["X_miRNA_proxy"], dtype=np.float32)
    proxy_names = adata.uns.get("miRNA_proxy_names", [f"mirna_{i}" for i in range(proxy_arr.shape[1])])
    proxy_df = pd.DataFrame(proxy_arr, index=adata.obs_names, columns=proxy_names)
    proxy_df.columns = [_norm_mir_name(c) for c in proxy_df.columns]
    
    emit_prior = np.asarray(adata.obs["emit_prior_01"], dtype=np.float32)
    risc_01 = np.asarray(adata.obs["risc_score_01"], dtype=np.float32)
    release_01 = np.asarray(adata.obs["release_score_01"], dtype=np.float32)
    sorting_01 = np.asarray(adata.obs["sorting_score_01"], dtype=np.float32)
    
    m2t = pd.read_csv(mir2tar_csv)
    
    mi_col = next((c for c in ["miRNA_mature", "miRNA", "miR", "mirna"] if c in m2t.columns), None)
    tg_col = next((c for c in ["target_gene", "target", "gene"] if c in m2t.columns), None)
    
    m2t["mi_norm"] = m2t[mi_col].astype(str).map(_norm_mir_name)
    m2t[tg_col] = m2t[tg_col].astype(str)
    
    m2t = m2t[m2t[tg_col].isin(adata.var_names)]
    m2t["target_idx"] = m2t[tg_col].map(adata.var_names.get_loc)
    
    m2t["conf_weight"] = m2t.apply(assign_confidence_weight, axis=1)
    m2t_best = m2t.groupby(["mi_norm", "target_idx"])["conf_weight"].max().reset_index()
    
    mir2info = {}
    for mir, grp in m2t_best.groupby("mi_norm"):
        valid = grp[grp["conf_weight"] > 0.05]
        t_idx = valid["target_idx"].tolist()
        t_w = valid["conf_weight"].tolist()
        if len(t_idx) >= config.MIN_TARGETS:
            mir2info[mir] = (np.array(t_idx, dtype=np.int32), np.array(t_w, dtype=np.float32))
    
    proxy_names = list(proxy_df.columns)

    keep = adata.uns.get("miRNA_proxy_keep", None)
    if keep is not None and len(keep) == len(proxy_names):
        kept_names = [m for m, k in zip(proxy_names, keep) if k]
    else:
        proxy_mat = np.asarray(adata.obsm["X_miRNA_proxy"])
        col_std = np.nanstd(proxy_mat, axis=0)
        std_min = getattr(config, "PROXY_STD_MIN", 1e-6)
        kept_names = [m for m, s in zip(proxy_names, col_std) if s > std_min]

    mir_candidates = [m for m in kept_names if m in mir2info]

    max_m = getattr(config, "MAX_MIRNA_CANDIDATES", None)
    if max_m is not None and len(mir_candidates) > max_m:
        qcov = adata.uns.get("miRNA_proxy_qcov", None)
        if qcov is not None and len(qcov) == len(proxy_names):
            cov_map = {m: float(c) for m, c in zip(proxy_names, qcov)}
            mir_candidates.sort(key=lambda m: cov_map.get(m, 0.0), reverse=True)
        mir_candidates = mir_candidates[:max_m]

    print(f"  miRNA candidates: {len(mir_candidates)}")
    
    X = adata.X.tocsr() if sp.issparse(adata.X) else csr_matrix(adata.X)
    
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    n_cells_total, n_genes_total = X_dense.shape
    rank_norm = np.zeros((n_cells_total, n_genes_total), dtype=np.float32)
    for ci in range(n_cells_total):
        rank_norm[ci] = np.argsort(np.argsort(-X_dense[ci])).astype(np.float32) / max(1, n_genes_total - 1)
    print(f"  Rank matrix computed: {rank_norm.shape}")
    
    
    gene_mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    NBINS = 30
    bins = pd.qcut(gene_mean, q=NBINS, labels=False, duplicates="drop")
    bins = np.asarray(bins)
    
    genes_by_bin = {}
    for b in np.unique(bins):
        genes_by_bin[int(b)] = np.where(bins == b)[0]
    
    rng = np.random.default_rng(config.SYNTH_RANDOM_SEED)
    
    def sample_matched_controls(target_idx, k):
        target_idx = np.asarray(target_idx, dtype=np.int32)
        chosen = []
        for g in target_idx:
            b = int(bins[g])
            pool = genes_by_bin.get(b, np.array([]))
            pool = pool[~np.isin(pool, target_idx)]
            if pool.size == 0:
                continue
            chosen.append(int(rng.choice(pool, size=1)[0]))
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            pool = np.setdiff1d(np.arange(adata.n_vars, dtype=np.int32), target_idx)
            if pool.size > 0:
                extra = rng.choice(pool, size=min(k - len(chosen), pool.size), replace=False)
                chosen.extend([int(x) for x in extra])
        return np.asarray(chosen[:k], dtype=np.int32)
    
    S = min(config.TOP_S_SENDERS, adata.n_obs)
    sender_idx_global = np.argpartition(-emit_prior, S-1)[:S]
    sender_idx_global = sender_idx_global[emit_prior[sender_idx_global] > 0].astype(np.int32)
    
    miRNA_proxy_mat = proxy_arr
    mir_name_to_idx = {_norm_mir_name(n): i for i, n in enumerate(proxy_names)}
    
    row_list, col_list, feat_list, ligidx_list = [], [], [], []
    
    print(f"  Building edges for {len(mir_candidates)} miRNAs...")
    
    for mi_idx, m in enumerate(mir_candidates):
        tgt_idx, tgt_w = mir2info.get(m, (np.array([]), np.array([])))
        if len(tgt_idx) < config.MIN_TARGETS:
            continue
        
        sw = float(tgt_w.sum())
        if sw <= 1e-6:
            continue
        
        proxy_idx = mir_name_to_idx.get(m, None)
        if proxy_idx is not None:
            mir_proxy_expr = miRNA_proxy_mat[:, proxy_idx]
            mir_proxy_01 = z01(mir_proxy_expr)
            sender_score = emit_prior * mir_proxy_01
            
            nonzero_count = np.sum(sender_score > 0)
            if nonzero_count == 0:
                sender_idx = sender_idx_global
                sender_score_use = emit_prior
                mir_proxy_01_use = np.ones(adata.n_obs, dtype=np.float32)
            else:
                S_eff = min(S, nonzero_count)
                sender_idx = np.argpartition(-sender_score, S_eff-1)[:S_eff]
                sender_idx = sender_idx[sender_score[sender_idx] > 0].astype(np.int32)
                sender_score_use = sender_score
                mir_proxy_01_use = mir_proxy_01
        else:
            sender_idx = sender_idx_global
            sender_score_use = emit_prior
            mir_proxy_01_use = np.ones(adata.n_obs, dtype=np.float32)
        
        if sender_idx.size == 0:
            continue
        
        tmod_t = np.asarray((X[:, tgt_idx] @ tgt_w) / sw).ravel().astype(np.float32)
        
        ctrl_idx = sample_matched_controls(tgt_idx, k=len(tgt_idx))
        if len(ctrl_idx) > 0:
            tmod_c = np.asarray((X[:, ctrl_idx] @ tgt_w[:len(ctrl_idx)]) / sw).ravel().astype(np.float32)
        else:
            tmod_c = np.zeros_like(tmod_t)
        
        tmod_adj = (tmod_t - tmod_c).astype(np.float32)
        repress_orig = z01(-tmod_adj)
        
        rank_repress = np.asarray(
            (rank_norm[:, tgt_idx] * tgt_w[np.newaxis, :]).sum(axis=1) / sw
        ).ravel().astype(np.float32)
        repress_rank = z01(rank_repress)
        
        repress_01 = np.sqrt(repress_orig * repress_rank).astype(np.float32)
        
        absorb_all = repress_01 * risc_01
        
        R = min(config.TOP_R_RECEIVERS, adata.n_obs)
        nonzero_absorb = np.sum(absorb_all > 0)
        if nonzero_absorb == 0:
            continue
        R_eff = min(R, nonzero_absorb)
        recv_idx = np.argpartition(-absorb_all, R_eff-1)[:R_eff]
        recv_idx = recv_idx[absorb_all[recv_idx] > 0].astype(np.int32)
        
        if recv_idx.size == 0:
            continue
        
        recv_sorted = recv_idx[np.argsort(-absorb_all[recv_idx])]
        L = min(config.TOP_L_PER_SENDER, recv_sorted.size)
        recv_topL = recv_sorted[:L]
        
        for a in sender_idx:
            ea = float(sender_score_use[a])
            if ea <= 0:
                continue
            
            w_vec = ea * absorb_all[recv_topL]
            pass_mask = w_vec > config.COEXPR_THRESHOLD
            if not np.any(pass_mask):
                continue
            
            bs = recv_topL[pass_mask]
            ws = w_vec[pass_mask]
            
            for b, wv in zip(bs, ws):
                if a == b:
                    continue
                row_list.append(int(a))
                col_list.append(int(b))
                feat_list.append([
                    float(wv),
                    float(emit_prior[a]),
                    float(repress_01[b]),
                    float(risc_01[b]),
                    float(tmod_adj[b]),
                    float(mir_proxy_01_use[a]),
                ])
                ligidx_list.append(int(mi_idx))
        
        if (mi_idx + 1) % 100 == 0:
            print(f"    Processed {mi_idx + 1}/{len(mir_candidates)} miRNAs, edges: {len(row_list)}")
    
    if len(row_list) > 0:
        row_col = np.vstack([row_list, col_list]).T.astype(np.int32)
        edge_feat = np.asarray(feat_list, dtype=np.float32)
        ligidx = np.asarray(ligidx_list, dtype=np.int32)
    else:
        row_col = np.zeros((0, 2), dtype=np.int32)
        edge_feat = np.zeros((0, 6), dtype=np.float32)
        ligidx = np.zeros((0,), dtype=np.int32)
    
    node_feat = np.vstack([release_01, sorting_01, risc_01]).T.astype(np.float32)
    
    payload = {
        "row_col": row_col,
        "edge_feat": edge_feat,
        "ligidx": ligidx,
        "num_nodes": int(adata.n_obs),
        "release_01": release_01,
        "sorting_01": sorting_01,
        "risc_01": risc_01,
        "node_feat": node_feat,
        "cell_index_map": {c: i for i, c in enumerate(adata.obs_names)},
        "mirna_list": mir_candidates,
        "miRNA_proxy_mat": miRNA_proxy_mat,
        "miRNA_proxy_names": proxy_names,
    }
    
    print(f"\n  Graph built in {time.time() - t0:.1f}s")
    print(f"  Nodes: {payload['num_nodes']}")
    print(f"  Edges: {row_col.shape[0]}")
    print(f"  Edge features: {edge_feat.shape[1]}")
    print(f"  miRNAs: {len(mir_candidates)}")
    
    return payload


if TORCH_AVAILABLE:
    
    def set_seed(seed):
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    class MiRNAAwareEncoder(nn.Module):
        
        def __init__(self, in_channels, hidden_channels, heads,
                     edge_cont_dim, num_mirna, mir_emb_dim=16):
            super().__init__()
            self.mir_emb_dim = int(mir_emb_dim)
            self.edge_cont_dim = int(edge_cont_dim)
            self.num_mirna = int(num_mirna)
            
            self.mir_emb = nn.Embedding(self.num_mirna, self.mir_emb_dim)
            edge_dim = self.edge_cont_dim + self.mir_emb_dim
            
            self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                         concat=False, edge_dim=edge_dim)
            self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, 
                                         concat=False, edge_dim=edge_dim)
            
            self.prelu = nn.PReLU(hidden_channels)
            
            self.attention_scores_mine_l1 = None
            self.attention_scores_mine = None
        
        def forward(self, data):
            if not hasattr(data, "edge_mir"):
                raise AttributeError("Data missing edge_mir")
            
            e_m = self.mir_emb(data.edge_mir)
            edge_attr_aug = torch.cat([data.edge_attr, e_m], dim=-1)
            
            x, (ei1, a1) = self.conv1(
                data.x, data.edge_index, edge_attr=edge_attr_aug, 
                return_attention_weights=True
            )
            self.attention_scores_mine_l1 = (ei1, a1)
            
            x, (ei2, a2) = self.conv2(
                x, data.edge_index, edge_attr=edge_attr_aug,
                return_attention_weights=True
            )
            self.attention_scores_mine = (ei2, a2)
            
            return self.prelu(x)
    
    
    def corruption(data, *args, **kwargs):
        node_perm = torch.randperm(data.x.size(0), device=data.x.device)
        edge_perm = torch.randperm(data.edge_attr.size(0), device=data.edge_attr.device)
        neg = Data(
            x=data.x[node_perm],
            edge_index=data.edge_index,
            edge_attr=data.edge_attr[edge_perm],
        )
        if hasattr(data, "edge_mir"):
            neg.edge_mir = data.edge_mir
        if hasattr(data, "global_id"):
            neg.global_id = data.global_id
        if hasattr(data, "num_mirna"):
            neg.num_mirna = data.num_mirna
        return neg
    
    
    def get_graph_loader(payload):
        row_col = payload['row_col']
        num_nodes = int(payload['num_nodes'])
        X_data = payload['node_feat']
        edge_feat = payload['edge_feat']
        ligidx = payload['ligidx']
        
        edge_index = torch.tensor(np.asarray(row_col), dtype=torch.long).T.contiguous()
        x = torch.tensor(np.asarray(X_data), dtype=torch.float32)
        edge_attr = torch.tensor(np.asarray(edge_feat), dtype=torch.float32)
        edge_mir = torch.tensor(np.asarray(ligidx), dtype=torch.long)
        
        num_mirna = int(edge_mir.max().item()) + 1 if edge_mir.numel() > 0 else 1
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.edge_mir = edge_mir
        graph.global_id = torch.arange(num_nodes)
        graph.num_mirna = num_mirna
        
        receivers = torch.unique(graph.edge_index[1])
        
        data_loader = NeighborLoader(
            graph,
            input_nodes=receivers,
            num_neighbors=[10, 10],
            batch_size=1024,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        in_channels = graph.x.size(1)
        edge_cont_dim = graph.edge_attr.size(1)
        
        return data_loader, graph, in_channels, edge_cont_dim, num_mirna
    
    
    def train_dgi_model(payload, config: Config, model_path: str):
        print("\n" + "=" * 60)
        print("STEP 4: Training DGI Model")
        print("=" * 60)
        
        set_seed(config.SYNTH_RANDOM_SEED)
        
        device = torch.device(config.DEVICE)
        print(f"  Using device: {device}")
        
        data_loader, graph, in_channels, edge_cont_dim, num_mirna = get_graph_loader(payload)
        
        print(f"  Graph: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
        print(f"  Node features: {in_channels}, Edge features: {edge_cont_dim}")
        print(f"  miRNAs: {num_mirna}")
        
        DGI_model = DeepGraphInfomax(
            hidden_channels=config.HIDDEN_DIM,
            encoder=MiRNAAwareEncoder(
                in_channels=in_channels,
                hidden_channels=config.HIDDEN_DIM,
                heads=config.NUM_HEADS,
                edge_cont_dim=edge_cont_dim,
                num_mirna=num_mirna,
                mir_emb_dim=config.MIR_EMB_DIM,
            ),
            summary=lambda z, *a, **k: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        ).to(device)
        
        optimizer = torch.optim.Adam(DGI_model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LEARNING_RATE * 0.01
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        tau = 0.5
        
        os.makedirs(model_path, exist_ok=True)
        ckpt_path = os.path.join(model_path, "DGI_model.pth.tar")
        
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_path, "DGI_init.pth.tar"))
        
        min_loss = float('inf')
        start_time = time.time()
        
        print(f"\n  Training for {config.NUM_EPOCHS} epochs...")
        
        for epoch in range(config.NUM_EPOCHS):
            DGI_model.train()
            epoch_losses = []
            
            for sub_data in data_loader:
                if sub_data.num_nodes == 0 or sub_data.edge_index.numel() == 0:
                    continue
                
                sub_data = sub_data.to(device, non_blocking=True)
                
                pos_z, neg_z, summary = DGI_model(data=sub_data)
                
                pos = torch.nn.functional.normalize(pos_z, dim=-1)
                neg = torch.nn.functional.normalize(neg_z, dim=-1)
                s = torch.nn.functional.normalize(summary, dim=-1)
                
                pos_logit = (pos * s).sum(dim=-1) / tau
                neg_logit = (neg * s).sum(dim=-1) / tau
                
                loss = 0.5 * (
                    criterion(pos_logit, torch.ones_like(pos_logit)) +
                    criterion(neg_logit, torch.zeros_like(neg_logit))
                )
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DGI_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if (epoch + 1) % config.PRINT_EVERY == 0:
                current_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch + 1:4d}, Loss: {current_loss:.4f}, LR: {current_lr:.6f}")
                
                if current_loss < min_loss:
                    min_loss = current_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': DGI_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': min_loss,
                    }, ckpt_path)
            
            scheduler.step()
        
        train_time = time.time() - start_time
        print(f"\n  Training completed in {train_time:.1f}s")
        print(f"  Best loss: {min_loss:.4f}")
        print(f"  Model saved to: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.eval()
        
        return DGI_model, graph, in_channels, edge_cont_dim, num_mirna


if TORCH_AVAILABLE:
    
    def partition_directed_graph_as_undirected(graph: Data, num_parts: int, overlap_hops: int):
        print("  Partitioning graph...")
        graph.orig_idx = torch.arange(graph.num_nodes)
        
        undirected_edge_index = to_undirected(graph.edge_index)
        graph_undirected = Data(x=graph.x, edge_index=undirected_edge_index, orig_idx=graph.orig_idx)
        
        cluster_data = ClusterData(graph_undirected, num_parts=num_parts, recursive=True)
        print(f"    METIS created {len(cluster_data)} partitions")
        
        subgraph_list = []
        for i, sub_data in enumerate(cluster_data):
            base_node_ids = sub_data.orig_idx
            
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                base_node_ids,
                num_hops=overlap_hops,
                edge_index=graph.edge_index,
                relabel_nodes=True,
                num_nodes=graph.num_nodes
            )
            
            sub_x = graph.x[subset]
            
            sub_edge_attr = None
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                sub_edge_attr = graph.edge_attr[edge_mask]
            
            sub_edge_mir = None
            if hasattr(graph, "edge_mir"):
                sub_edge_mir = graph.edge_mir[edge_mask]
            
            subgraph = Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
            subgraph.orig_idx = subset
            
            if sub_edge_mir is not None:
                subgraph.edge_mir = sub_edge_mir
            
            if hasattr(graph, "num_mirna"):
                subgraph.num_mirna = graph.num_mirna
            
            subgraph_list.append(subgraph)
        
        print(f"    Generated {len(subgraph_list)} subgraphs with {overlap_hops}-hop overlap")
        return subgraph_list
    
    
    def run_inference(model, graph, subgraph_list, config: Config, output_path: str):
        print("\n" + "=" * 60)
        print("STEP 5: Running Inference")
        print("=" * 60)
        
        device = torch.device(config.DEVICE)
        model.eval()
        
        global_embeddings = torch.zeros((graph.num_nodes, config.HIDDEN_DIM), device=device)
        global_counts = torch.zeros((graph.num_nodes,), device=device)
        
        edge_attn_l1 = defaultdict(list)
        edge_attn_l2 = defaultdict(list)
        
        print(f"  Processing {len(subgraph_list)} subgraphs...")
        
        with torch.no_grad():
            for i, sub_data in enumerate(subgraph_list):
                if (i + 1) % 10 == 0:
                    print(f"    Subgraph {i+1}/{len(subgraph_list)}")
                
                if sub_data.num_nodes == 0 or sub_data.edge_index.numel() == 0:
                    continue
                
                sub_data = sub_data.to(device)
                
                pos_z, _, _ = model(sub_data)
                
                orig_ids = sub_data.orig_idx.to(device)
                global_embeddings[orig_ids] += pos_z
                global_counts[orig_ids] += 1
                
                (ei1, a1) = model.encoder.attention_scores_mine_l1
                (ei2, a2) = model.encoder.attention_scores_mine
                
                orig_ids_cpu = orig_ids.cpu()
                ei1_cpu = ei1.cpu()
                a1_cpu = a1.cpu()
                ei2_cpu = ei2.cpu()
                a2_cpu = a2.cpu()
                
                a1_avg = a1_cpu.mean(dim=-1)
                a2_avg = a2_cpu.mean(dim=-1)
                
                edge_mir_cpu = sub_data.edge_mir.cpu() if hasattr(sub_data, "edge_mir") else None
                
                for e_ind in range(ei1_cpu.size(1)):
                    loc_u = ei1_cpu[0, e_ind].item()
                    loc_v = ei1_cpu[1, e_ind].item()
                    gu = orig_ids_cpu[loc_u].item()
                    gv = orig_ids_cpu[loc_v].item()
                    
                    if edge_mir_cpu is not None and e_ind < edge_mir_cpu.numel():
                        mir = int(edge_mir_cpu[e_ind].item())
                    else:
                        mir = -1
                    if mir >= 0:
                        edge_attn_l1[(gu, gv, mir)].append(float(a1_avg[e_ind].item()))
                
                for e_ind in range(ei2_cpu.size(1)):
                    loc_u = ei2_cpu[0, e_ind].item()
                    loc_v = ei2_cpu[1, e_ind].item()
                    gu = orig_ids_cpu[loc_u].item()
                    gv = orig_ids_cpu[loc_v].item()
                    
                    if edge_mir_cpu is not None and e_ind < edge_mir_cpu.numel():
                        mir = int(edge_mir_cpu[e_ind].item())
                    else:
                        mir = -1
                    if mir >= 0:
                        edge_attn_l2[(gu, gv, mir)].append(float(a2_avg[e_ind].item()))
        
        print("  Aggregating results...")
        
        mask = global_counts > 0
        global_embeddings[mask] /= global_counts[mask].unsqueeze(1)
        global_embeddings[~mask] = 0
        
        final_attn_l1 = {k: float(np.mean(v)) for k, v in edge_attn_l1.items()}
        final_attn_l2 = {k: float(np.mean(v)) for k, v in edge_attn_l2.items()}
        
        os.makedirs(output_path, exist_ok=True)
        
        emb = global_embeddings.cpu().numpy()
        embed_path = os.path.join(output_path, "node_embeddings.pkl.gz")
        with gzip.open(embed_path, "wb") as fp:
            pickle.dump(emb, fp)
        print(f"  Embeddings saved to: {embed_path}")
        
        all_edges = []
        all_keys = set(final_attn_l1.keys()).union(set(final_attn_l2.keys()))
        for (u, v, mir) in all_keys:
            l1 = final_attn_l1.get((u, v, mir), 0.0)
            l2 = final_attn_l2.get((u, v, mir), 0.0)
            all_edges.append([u, v, mir, l1, l2])
        
        attn_path = os.path.join(output_path, "edge_attention.pkl.gz")
        with gzip.open(attn_path, "wb") as fp:
            pickle.dump(all_edges, fp)
        print(f"  Attention saved to: {attn_path}")
        
        return emb, all_edges, attn_path
    
    
    def filter_top_attention_edges(attention_file: str, retain_percent: float, out_file: str):
        print(f"\n  Filtering top {retain_percent*100:.1f}% edges...")
        
        with gzip.open(attention_file, "rb") as fp:
            edge_list = pickle.load(fp)
        
        total_edges = len(edge_list)
        if total_edges == 0:
            print("    Warning: No edges")
            return []
        
        edge_list.sort(key=lambda e: e[3], reverse=True)
        
        num_keep = max(1, int(total_edges * retain_percent))
        filtered = edge_list[:num_keep]
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wb") as fp:
            pickle.dump(filtered, fp)
        
        print(f"    Kept {len(filtered)} / {total_edges} edges")
        print(f"    Saved to: {out_file}")
        
        return filtered


"""
Standardized export functions for downstream analysis, visualization, and reproducibility.

These functions convert internal edge representations to well-documented DataFrames
that separate attention scores from prior weights, making interpretation clearer.

API:
    - edges_to_mirna_edge_df(): Convert edge list to detailed miRNA-level DataFrame
    - aggregate_to_cellpair_df(): Aggregate to cell-pair level for network analysis
"""


def edges_to_mirna_edge_df(
    edges: List[List],
    adata: ad.AnnData,
    payload: Dict,
    include_cell_names: bool = True,
    include_cell_types: bool = True,
) -> pd.DataFrame:
    if edges is None or len(edges) == 0:
        print("[WARNING] edges_to_mirna_edge_df: Empty edge list provided")
        return pd.DataFrame()
    
    mirna_list = payload.get('mirna_list', [])
    row_col = payload.get('row_col', np.array([]))
    edge_feat = payload.get('edge_feat', np.array([]))
    ligidx = payload.get('ligidx', np.array([]))
    
    edge_feat_lookup = {}
    if len(row_col) > 0 and len(edge_feat) > 0:
        for i in range(len(row_col)):
            key = (int(row_col[i, 0]), int(row_col[i, 1]), int(ligidx[i]))
            edge_feat_lookup[key] = edge_feat[i]
    
    cell_names = list(adata.obs_names)
    
    has_cell_type = 'cell_type' in adata.obs.columns
    if has_cell_type:
        cell_types = adata.obs['cell_type'].values
    
    records = []
    for edge in edges:
        u, v, mir_idx, attn_l1, attn_l2 = edge[0], edge[1], edge[2], edge[3], edge[4]
        u, v, mir_idx = int(u), int(v), int(mir_idx)
        
        if 0 <= mir_idx < len(mirna_list):
            mirna_name = mirna_list[mir_idx]
        else:
            mirna_name = f"unknown_{mir_idx}"
        
        attn_avg = 0.5 * (float(attn_l1) + float(attn_l2))
        
        feat_key = (u, v, mir_idx)
        if feat_key in edge_feat_lookup:
            ef = edge_feat_lookup[feat_key]
            prior_w = float(ef[0])
            emit_prior = float(ef[1])
            repress_01 = float(ef[2])
            risc_01 = float(ef[3])
            tmod_adj = float(ef[4])
            mir_proxy_01 = float(ef[5])
        else:
            prior_w = np.nan
            emit_prior = float(payload.get('release_01', np.zeros(1))[u]) if u < len(payload.get('release_01', [])) else np.nan
            repress_01 = np.nan
            risc_01 = float(payload.get('risc_01', np.zeros(1))[v]) if v < len(payload.get('risc_01', [])) else np.nan
            tmod_adj = np.nan
            mir_proxy_01 = np.nan
        
        _proxy = mir_proxy_01 if (not np.isnan(mir_proxy_01)) else 0.0
        _repress = repress_01 if (not np.isnan(repress_01)) else 0.0
        _emit = emit_prior if (not np.isnan(emit_prior)) else 0.0
        _risc = risc_01 if (not np.isnan(risc_01)) else 0.0
        
        match_score = _proxy * _repress
        
        machinery_gate = np.sqrt(_emit * _risc) if (_emit > 0 and _risc > 0) else 0.0
        
        n_edges_total = len(edges)
        avg_degree = max(1, n_edges_total / max(1, payload.get('num_nodes', 1)))
        _attn_scale = avg_degree / 2.0
        attn_bonus = _attn_scale * attn_avg
        
        score = match_score * machinery_gate * (1.0 + attn_bonus)
        
        record = {
            'sender_cell': u,
            'receiver_cell': v,
            'mirna': mirna_name,
            'mirna_idx': mir_idx,
            'score': score,
            'match_score': match_score,
            'attn_l1': float(attn_l1),
            'attn_l2': float(attn_l2),
            'prior_w': prior_w,
            'emit_prior': emit_prior,
            'mir_proxy_01': mir_proxy_01,
            'risc_01': risc_01,
            'repress_01': repress_01,
            'tmod_adj': tmod_adj,
        }
        
        if include_cell_names:
            record['sender_name'] = cell_names[u] if u < len(cell_names) else f"cell_{u}"
            record['receiver_name'] = cell_names[v] if v < len(cell_names) else f"cell_{v}"
        
        if include_cell_types and has_cell_type:
            record['sender_type'] = cell_types[u] if u < len(cell_types) else "unknown"
            record['receiver_type'] = cell_types[v] if v < len(cell_types) else "unknown"
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    col_order = [
        'sender_cell', 'receiver_cell', 'mirna', 'mirna_idx',
        'score', 'match_score', 'attn_l1', 'attn_l2', 'prior_w',
        'emit_prior', 'mir_proxy_01', 'risc_01', 'repress_01', 'tmod_adj'
    ]
    if include_cell_names:
        col_order.extend(['sender_name', 'receiver_name'])
    if include_cell_types and has_cell_type:
        col_order.extend(['sender_type', 'receiver_type'])
    
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    
    print(f"  [edges_to_mirna_edge_df] Created DataFrame with {len(df)} edges, {df['mirna'].nunique()} unique miRNAs")
    
    return df


def aggregate_to_cellpair_df(
    mirna_edge_df: pd.DataFrame,
    agg: str = "sum",
    topk: int = 5,
    include_mirna_list: bool = True,
) -> pd.DataFrame:
    if mirna_edge_df is None or len(mirna_edge_df) == 0:
        print("[WARNING] aggregate_to_cellpair_df: Empty DataFrame provided")
        return pd.DataFrame()
    
    valid_aggs = {"sum", "max", "mean", "topk_sum", "topk_mean"}
    if agg not in valid_aggs:
        raise ValueError(f"agg must be one of {valid_aggs}, got '{agg}'")
    
    group_cols = ['sender_cell', 'receiver_cell']
    
    has_sender_name = 'sender_name' in mirna_edge_df.columns
    has_receiver_name = 'receiver_name' in mirna_edge_df.columns
    has_sender_type = 'sender_type' in mirna_edge_df.columns
    has_receiver_type = 'receiver_type' in mirna_edge_df.columns
    
    def agg_func(grp):
        scores = grp['score'].values
        mirnas = grp['mirna'].values
        
        sort_idx = np.argsort(-scores)
        scores_sorted = scores[sort_idx]
        mirnas_sorted = mirnas[sort_idx]
        
        if agg == "sum":
            agg_score = scores.sum()
        elif agg == "max":
            agg_score = scores.max()
        elif agg == "mean":
            agg_score = scores.mean()
        elif agg == "topk_sum":
            k = min(topk, len(scores))
            agg_score = scores_sorted[:k].sum()
        elif agg == "topk_mean":
            k = min(topk, len(scores))
            agg_score = scores_sorted[:k].mean()
        else:
            agg_score = scores.sum()
        
        result = {
            'score': agg_score,
            'n_mirnas': len(scores),
            'top_mirna': mirnas_sorted[0] if len(mirnas_sorted) > 0 else "",
        }
        
        if include_mirna_list:
            n_show = min(10, len(mirnas_sorted))
            result['mirna_list'] = ",".join(mirnas_sorted[:n_show])
        
        if has_sender_name:
            result['sender_name'] = grp['sender_name'].iloc[0]
        if has_receiver_name:
            result['receiver_name'] = grp['receiver_name'].iloc[0]
        if has_sender_type:
            result['sender_type'] = grp['sender_type'].iloc[0]
        if has_receiver_type:
            result['receiver_type'] = grp['receiver_type'].iloc[0]
        
        return pd.Series(result)
    
    result_df = mirna_edge_df.groupby(group_cols, as_index=False).apply(agg_func, include_groups=False)
    result_df = result_df.reset_index(drop=True)
    
    col_order = ['sender_cell', 'receiver_cell', 'score', 'n_mirnas', 'top_mirna']
    if include_mirna_list:
        col_order.append('mirna_list')
    if has_sender_name:
        col_order.append('sender_name')
    if has_receiver_name:
        col_order.append('receiver_name')
    if has_sender_type:
        col_order.append('sender_type')
    if has_receiver_type:
        col_order.append('receiver_type')
    
    col_order = [c for c in col_order if c in result_df.columns]
    result_df = result_df[col_order]
    
    result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    print(f"  [aggregate_to_cellpair_df] Aggregated to {len(result_df)} cell pairs (agg={agg})")
    
    return result_df


def export_communication_results(
    edges: List[List],
    adata: ad.AnnData,
    payload: Dict,
    output_dir: str,
    prefix: str = "mirage",
    formats: List[str] = ["csv", "parquet"],
    agg_methods: List[str] = ["max", "sum", "topk_sum"],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    
    print("\n" + "=" * 60)
    print("EXPORT: Standardized Communication Results")
    print("=" * 60)
    
    print("\n[1] Generating miRNA-level edge table...")
    edge_df = edges_to_mirna_edge_df(edges, adata, payload)
    
    if len(edge_df) > 0:
        for fmt in formats:
            fname = f"{prefix}_mirna_edges.{fmt}"
            fpath = os.path.join(output_dir, fname)
            
            if fmt == "csv":
                edge_df.to_csv(fpath, index=False)
            elif fmt == "tsv":
                edge_df.to_csv(fpath, index=False, sep='\t')
            elif fmt == "parquet":
                try:
                    edge_df.to_parquet(fpath, index=False)
                except ImportError:
                    print(f"    [SKIP] parquet format requires pyarrow")
                    continue
            elif fmt == "xlsx":
                try:
                    edge_df.to_excel(fpath, index=False)
                except ImportError:
                    print(f"    [SKIP] xlsx format requires openpyxl")
                    continue
            
            output_paths[f"mirna_edges_{fmt}"] = fpath
            print(f"    Saved: {fpath}")
    
    print("\n[2] Generating cell-pair level tables...")
    for agg in agg_methods:
        cellpair_df = aggregate_to_cellpair_df(edge_df, agg=agg)
        
        if len(cellpair_df) > 0:
            for fmt in formats:
                fname = f"{prefix}_cellpair_{agg}.{fmt}"
                fpath = os.path.join(output_dir, fname)
                
                if fmt == "csv":
                    cellpair_df.to_csv(fpath, index=False)
                elif fmt == "tsv":
                    cellpair_df.to_csv(fpath, index=False, sep='\t')
                elif fmt == "parquet":
                    try:
                        cellpair_df.to_parquet(fpath, index=False)
                    except ImportError:
                        continue
                elif fmt == "xlsx":
                    try:
                        cellpair_df.to_excel(fpath, index=False)
                    except ImportError:
                        continue
                
                output_paths[f"cellpair_{agg}_{fmt}"] = fpath
                print(f"    Saved: {fpath}")
    
    if 'sender_type' in edge_df.columns and 'receiver_type' in edge_df.columns:
        print("\n[3] Generating cell-type communication matrix...")
        
        cellpair_sum = aggregate_to_cellpair_df(edge_df, agg="sum")
        ct_matrix = cellpair_sum.groupby(['sender_type', 'receiver_type'])['score'].sum().unstack(fill_value=0)
        
        fname = f"{prefix}_celltype_matrix.csv"
        fpath = os.path.join(output_dir, fname)
        ct_matrix.to_csv(fpath)
        output_paths["celltype_matrix"] = fpath
        print(f"    Saved: {fpath}")
    
    print(f"\n  Export complete: {len(output_paths)} files generated")
    
    return output_paths


def evaluate_mirna_proxy(predicted_proxy, ground_truth, mirna, n_cells):
    if mirna not in ground_truth.get('mirna_activity', {}):
        return {'error': f'{mirna} not found'}
    
    gt = ground_truth['mirna_activity'][mirna]
    true_labels = np.zeros(n_cells)
    for idx in gt.get('sender_cells', []):
        if idx < n_cells:
            true_labels[idx] = 1
    
    if len(np.unique(true_labels)) < 2:
        return {'error': 'Single class'}
    
    return {
        'auroc': roc_auc_score(true_labels, predicted_proxy),
        'auprc': average_precision_score(true_labels, predicted_proxy),
        'pearson_r': pearsonr(predicted_proxy, true_labels)[0]
    }


def evaluate_communication(predicted_df, gt_labels_df, top_k=100):
    merged = predicted_df.merge(
        gt_labels_df[['sender_cell', 'receiver_cell', 'communication']],
        on=['sender_cell', 'receiver_cell']
    )
    
    if len(merged) == 0:
        return {'error': 'No overlap'}
    
    y_true = merged['communication'].values
    y_score = merged['score'].values
    
    if len(np.unique(y_true)) < 2:
        return {'error': 'Single class'}
    
    k = min(top_k, len(y_score))
    top_idx = np.argsort(y_score)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_idx] = 1
    
    return {
        'auroc': roc_auc_score(y_true, y_score),
        'auprc': average_precision_score(y_true, y_score),
        f'precision@{k}': precision_score(y_true, y_pred, zero_division=0),
        f'recall@{k}': recall_score(y_true, y_pred, zero_division=0),
        f'f1@{k}': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_with_ground_truth(adata, payload, ground_truth, gt_labels, mir2tar_csv, config: Config):
    print("\n" + "=" * 60)
    print("EVALUATION: Against Ground Truth")
    print("=" * 60)
    
    results = {
        'proxy_evaluation': {},
        'communication_evaluation': {},
        'pattern_recovery': {}
    }
    
    print("\n[1] Evaluating miRNA Proxy...")
    if "X_miRNA_proxy" in adata.obsm:
        proxy_mat = adata.obsm["X_miRNA_proxy"]
        proxy_names = adata.uns.get("miRNA_proxy_names", [])
        
        proxy_results = []
        for mirna in ground_truth.get('mirna_activity', {}).keys():
            norm_mirna = _norm_mir_name(mirna)
            matched_idx = None
            for i, pn in enumerate(proxy_names):
                if _norm_mir_name(pn) == norm_mirna or norm_mirna in _norm_mir_name(pn):
                    matched_idx = i
                    break
            
            if matched_idx is not None:
                pred = proxy_mat[:, matched_idx]
                eval_result = evaluate_mirna_proxy(pred, ground_truth, mirna, adata.n_obs)
                if 'error' not in eval_result:
                    eval_result['mirna'] = mirna
                    proxy_results.append(eval_result)
        
        if proxy_results:
            proxy_df = pd.DataFrame(proxy_results)
            results['proxy_evaluation'] = {
                'mean_auroc': proxy_df['auroc'].mean(),
                'mean_auprc': proxy_df['auprc'].mean(),
                'mean_pearson': proxy_df['pearson_r'].mean(),
                'n_evaluated': len(proxy_df),
                'details': proxy_df
            }
            print(f"    Evaluated {len(proxy_df)} miRNAs")
            print(f"    Mean AUROC: {proxy_df['auroc'].mean():.4f}")
            print(f"    Mean AUPRC: {proxy_df['auprc'].mean():.4f}")
    
    print("\n[2] Evaluating Communication Patterns...")
    if payload is not None and 'row_col' in payload:
        row_col = payload['row_col']
        edge_feat = payload['edge_feat']
        
        if len(row_col) > 0:
            pred_df = pd.DataFrame({
                'sender_cell': row_col[:, 0],
                'receiver_cell': row_col[:, 1],
                'score': edge_feat[:, 0]
            })
            
            eval_result = evaluate_communication(pred_df, gt_labels)
            if 'error' not in eval_result:
                results['communication_evaluation'] = eval_result
                print(f"    AUROC: {eval_result.get('auroc', 0):.4f}")
                print(f"    AUPRC: {eval_result.get('auprc', 0):.4f}")
    
    print("\n[3] Analyzing Pattern Recovery...")
    comm_pairs = ground_truth.get('communication_pairs', [])
    cell_types = adata.obs['cell_type'].values
    
    pattern_recovery = []
    for p in comm_pairs:
        sender_type = p['sender']
        receiver_type = p['receiver']
        
        sender_cells = np.where(cell_types == sender_type)[0]
        receiver_cells = np.where(cell_types == receiver_type)[0]
        
        if payload is not None and 'row_col' in payload:
            row_col = payload['row_col']
            sender_set = set(sender_cells)
            receiver_set = set(receiver_cells)
            
            n_edges = 0
            for edge in row_col:
                if edge[0] in sender_set and edge[1] in receiver_set:
                    n_edges += 1
            
            pattern_recovery.append({
                'sender': sender_type,
                'receiver': receiver_type,
                'expected_strength': p['strength'],
                'n_edges_found': n_edges,
                'n_sender_cells': len(sender_cells),
                'n_receiver_cells': len(receiver_cells),
                'recovered': n_edges > 0
            })
    
    if pattern_recovery:
        recovery_df = pd.DataFrame(pattern_recovery)
        n_recovered = recovery_df['recovered'].sum()
        results['pattern_recovery'] = {
            'n_patterns': len(recovery_df),
            'n_recovered': int(n_recovered),
            'recovery_rate': float(n_recovered / len(recovery_df)),
            'details': recovery_df
        }
        print(f"    Patterns recovered: {n_recovered}/{len(recovery_df)}")
        print(f"    Recovery rate: {n_recovered/len(recovery_df):.1%}")
    
    return results


def plot_communication_matrix(comm_matrix, output_path, filename="communication_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(comm_matrix, annot=True, fmt='.2f', cmap='Reds', 
                square=True, linewidths=0.5)
    plt.title('Ground Truth: Cell Communication Matrix')
    plt.xlabel('Receiver')
    plt.ylabel('Sender')
    plt.tight_layout()
    
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_data_overview(adata, output_path, filename="data_overview.png"):
    print("\n  Generating data overview plots...")
    
    adata_vis = adata.copy()
    sc.pp.normalize_total(adata_vis, target_sum=1e4)
    sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, n_top_genes=2000)
    sc.pp.pca(adata_vis)
    sc.pp.neighbors(adata_vis)
    sc.tl.umap(adata_vis)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sc.pl.umap(adata_vis, color='cell_type', ax=axes[0,0], show=False, title='Cell Types')
    
    if 'batch' in adata_vis.obs.columns:
        sc.pl.umap(adata_vis, color='batch', ax=axes[0,1], show=False, title='Batch')
    else:
        axes[0,1].text(0.5, 0.5, 'No batch info', ha='center', va='center')
        axes[0,1].set_title('Batch')
    
    expr = adata.X.toarray().flatten() if sp.issparse(adata.X) else adata.X.flatten()
    axes[0,2].hist(np.log1p(expr[expr > 0]), bins=50, density=True, alpha=0.7)
    axes[0,2].set_xlabel('log(Expr + 1)')
    axes[0,2].set_title('Expression Distribution')
    
    adata.obs['cell_type'].value_counts().plot(kind='bar', ax=axes[1,0], color='steelblue')
    axes[1,0].set_title('Cell Type Distribution')
    plt.setp(axes[1,0].get_xticklabels(), rotation=45, ha='right')
    
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    sparsity = 1 - (X > 0).sum(axis=1) / adata.n_vars
    axes[1,1].hist(sparsity, bins=30, alpha=0.7)
    axes[1,1].axvline(sparsity.mean(), color='red', linestyle='--', label=f'Mean={sparsity.mean():.3f}')
    axes[1,1].set_title('Sparsity per Cell')
    axes[1,1].legend()
    
    genes_per_cell = (X > 0).sum(axis=1)
    axes[1,2].hist(genes_per_cell, bins=30, alpha=0.7, color='green')
    axes[1,2].axvline(genes_per_cell.mean(), color='red', linestyle='--', label=f'Mean={genes_per_cell.mean():.0f}')
    axes[1,2].set_title('Genes Detected per Cell')
    axes[1,2].legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_proxy_validation(adata, mirna_db, ground_truth, output_path, filename="proxy_validation.png"):
    print("\n  Generating proxy validation plots...")
    
    comm_pairs = ground_truth.get('communication_pairs', [])
    if not comm_pairs:
        print("    No communication pairs to validate")
        return
    
    n_plots = min(4, len(comm_pairs))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, ex in enumerate(comm_pairs[:n_plots]):
        ax = axes[idx]
        
        targets = []
        for m in ex['mirnas']:
            targets.extend([t for t in mirna_db.get_targets(m) if t in adata.var_names])
        targets = list(set(targets))[:5]
        
        if not targets:
            ax.text(0.5, 0.5, 'No targets found', ha='center', va='center')
            ax.set_title(f"{ex['sender']} Ã¢â€ â€™ {ex['receiver']}")
            continue
        
        sender_mask = adata.obs['cell_type'] == ex['sender']
        receiver_mask = adata.obs['cell_type'] == ex['receiver']
        other_mask = ~sender_mask & ~receiver_mask
        
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        target_idx = [adata.var_names.get_loc(t) for t in targets]
        
        sender_expr = X[sender_mask][:, target_idx].mean(axis=0)
        receiver_expr = X[receiver_mask][:, target_idx].mean(axis=0)
        other_expr = X[other_mask][:, target_idx].mean(axis=0)
        
        x = np.arange(len(targets))
        w = 0.25
        
        ax.bar(x - w, sender_expr, w, label=f'{ex["sender"]} (Sender)')
        ax.bar(x, receiver_expr, w, label=f'{ex["receiver"]} (Receiver)', color='coral')
        ax.bar(x + w, other_expr, w, label='Other', color='gray', alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Expression')
        ax.set_title(f'{ex["sender"]} Ã¢â€ â€™ {ex["receiver"]}\n(Expect: Receiver < Other)')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_sender_receiver_scores(adata, output_path, filename="sender_receiver_scores.png"):
    print("\n  Generating sender/receiver score plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'release_score_01' in adata.obs.columns:
        df = adata.obs[['cell_type', 'release_score_01']].copy()
        ct_order = df.groupby('cell_type')['release_score_01'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x='cell_type', y='release_score_01', order=ct_order, ax=axes[0,0], palette='Set2')
        axes[0,0].set_title('Release Score (Sender) by Cell Type')
        plt.setp(axes[0,0].get_xticklabels(), rotation=45, ha='right')
    
    if 'risc_score_01' in adata.obs.columns:
        df = adata.obs[['cell_type', 'risc_score_01']].copy()
        ct_order = df.groupby('cell_type')['risc_score_01'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x='cell_type', y='risc_score_01', order=ct_order, ax=axes[0,1], palette='Set3')
        axes[0,1].set_title('RISC Score (Receiver) by Cell Type')
        plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha='right')
    
    if 'emit_prior_01' in adata.obs.columns and 'risc_score_01' in adata.obs.columns:
        for ct in adata.obs['cell_type'].unique()[:8]:
            mask = adata.obs['cell_type'] == ct
            axes[1,0].scatter(
                adata.obs.loc[mask, 'emit_prior_01'],
                adata.obs.loc[mask, 'risc_score_01'],
                alpha=0.5, s=20, label=ct
            )
        axes[1,0].set_xlabel('Emit Prior (Sender)')
        axes[1,0].set_ylabel('RISC Score (Receiver)')
        axes[1,0].set_title('Sender vs Receiver Scores')
        axes[1,0].legend(fontsize=8, loc='best')
    
    if 'emit_prior_01' in adata.obs.columns:
        axes[1,1].hist(adata.obs['emit_prior_01'], bins=30, alpha=0.5, label='Emit Prior', color='blue')
    if 'risc_score_01' in adata.obs.columns:
        axes[1,1].hist(adata.obs['risc_score_01'], bins=30, alpha=0.5, label='RISC Score', color='red')
    axes[1,1].set_xlabel('Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Score Distributions')
    axes[1,1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_graph_statistics(payload, output_path, filename="graph_statistics.png"):
    print("\n  Generating graph statistics plots...")
    
    if payload is None or 'row_col' not in payload:
        print("    No payload available")
        return
    
    row_col = payload['row_col']
    edge_feat = payload['edge_feat']
    ligidx = payload['ligidx']
    num_nodes = payload['num_nodes']
    mirna_list = payload.get('mirna_list', [])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    if len(edge_feat) > 0:
        w = edge_feat[:, 0]
        axes[0,0].hist(w, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Edge Weight')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Edge Weight Distribution')
    
    if len(ligidx) > 0:
        cnt_mi = np.bincount(ligidx, minlength=len(mirna_list))
        axes[0,1].hist(cnt_mi[cnt_mi > 0], bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Number of Edges')
        axes[0,1].set_ylabel('Number of miRNAs')
        axes[0,1].set_title('Edges per miRNA')
    
    if len(row_col) > 0:
        src = row_col[:, 0]
        dst = row_col[:, 1]
        out_deg = np.bincount(src, minlength=num_nodes)
        in_deg = np.bincount(dst, minlength=num_nodes)
        
        axes[0,2].hist(out_deg[out_deg > 0], bins=30, alpha=0.5, label='Out-degree', color='blue')
        axes[0,2].hist(in_deg[in_deg > 0], bins=30, alpha=0.5, label='In-degree', color='red')
        axes[0,2].set_xlabel('Degree')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Node Degree Distribution')
        axes[0,2].legend()
    
    if len(ligidx) > 0 and len(mirna_list) > 0:
        cnt_mi = np.bincount(ligidx, minlength=len(mirna_list))
        top_idx = np.argsort(-cnt_mi)[:15]
        top_names = [mirna_list[i] if i < len(mirna_list) else f'mir_{i}' for i in top_idx]
        axes[1,0].barh(range(len(top_idx)), cnt_mi[top_idx])
        axes[1,0].set_yticks(range(len(top_idx)))
        axes[1,0].set_yticklabels(top_names, fontsize=8)
        axes[1,0].set_xlabel('Number of Edges')
        axes[1,0].set_title('Top 15 miRNAs by Edge Count')
        axes[1,0].invert_yaxis()
    
    if len(edge_feat) > 0 and edge_feat.shape[1] >= 6:
        feat_names = ['w_base', 'emit_i', 'repress_j', 'risc_j', 'tmod_adj_j', 'proxy_i']
        corr = np.corrcoef(edge_feat.T)
        sns.heatmap(corr, annot=True, fmt='.2f', xticklabels=feat_names, 
                    yticklabels=feat_names, ax=axes[1,1], cmap='coolwarm', center=0)
        axes[1,1].set_title('Edge Feature Correlations')
    
    summary_text = f"""Graph Summary:
    Nodes: {num_nodes}
    Edges: {len(row_col)}
    miRNAs: {len(mirna_list)}
    Edge features: {edge_feat.shape[1] if len(edge_feat) > 0 else 0}
    
    Avg out-degree: {out_deg.mean():.2f}
    Avg in-degree: {in_deg.mean():.2f}
    Max out-degree: {out_deg.max()}
    Max in-degree: {in_deg.max()}
    """
    axes[1,2].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                   verticalalignment='center', transform=axes[1,2].transAxes)
    axes[1,2].axis('off')
    axes[1,2].set_title('Summary Statistics')
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_evaluation_results(eval_results, output_path, filename="evaluation_results.png"):
    print("\n  Generating evaluation results plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    if 'proxy_evaluation' in eval_results and 'details' in eval_results['proxy_evaluation']:
        proxy_df = eval_results['proxy_evaluation']['details']
        
        axes[0,0].scatter(proxy_df['auroc'], proxy_df['auprc'], alpha=0.6, s=50)
        axes[0,0].axhline(proxy_df['auprc'].mean(), color='red', linestyle='--', 
                          label=f"Mean AUPRC={proxy_df['auprc'].mean():.3f}")
        axes[0,0].axvline(proxy_df['auroc'].mean(), color='blue', linestyle='--',
                          label=f"Mean AUROC={proxy_df['auroc'].mean():.3f}")
        axes[0,0].set_xlabel('AUROC')
        axes[0,0].set_ylabel('AUPRC')
        axes[0,0].set_title('miRNA Proxy Performance')
        axes[0,0].legend()
        axes[0,0].set_xlim([0, 1])
        axes[0,0].set_ylim([0, 1])
    
    if 'communication_evaluation' in eval_results and 'auroc' in eval_results['communication_evaluation']:
        comm_eval = eval_results['communication_evaluation']
        metrics = ['auroc', 'auprc']
        values = [comm_eval.get(m, 0) for m in metrics]
        
        bars = axes[0,1].bar(metrics, values, color=['steelblue', 'coral'])
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_title('Communication Prediction Performance')
        axes[0,1].set_ylim([0, 1])
        for bar, val in zip(bars, values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{val:.3f}', ha='center', fontsize=10)
    
    if 'pattern_recovery' in eval_results and 'details' in eval_results['pattern_recovery']:
        recovery_df = eval_results['pattern_recovery']['details']
        
        colors = ['green' if r else 'red' for r in recovery_df['recovered']]
        x = range(len(recovery_df))
        axes[1,0].bar(x, recovery_df['n_edges_found'], color=colors, alpha=0.7)
        axes[1,0].set_xticks(x)
        labels = [f"{r['sender'][:3]}Ã¢â€ â€™{r['receiver'][:3]}" for _, r in recovery_df.iterrows()]
        axes[1,0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1,0].set_ylabel('Edges Found')
        axes[1,0].set_title(f"Pattern Recovery (Rate: {eval_results['pattern_recovery']['recovery_rate']:.1%})")
    
    summary_lines = ["EVALUATION SUMMARY", "="*40]
    
    if 'proxy_evaluation' in eval_results and 'mean_auroc' in eval_results['proxy_evaluation']:
        pe = eval_results['proxy_evaluation']
        summary_lines.extend([
            f"",
            f"miRNA Proxy Prediction:",
            f"  Mean AUROC: {pe['mean_auroc']:.4f}",
            f"  Mean AUPRC: {pe['mean_auprc']:.4f}",
            f"  miRNAs evaluated: {pe['n_evaluated']}",
        ])
    
    if 'communication_evaluation' in eval_results and 'auroc' in eval_results['communication_evaluation']:
        ce = eval_results['communication_evaluation']
        summary_lines.extend([
            f"",
            f"Communication Prediction:",
            f"  AUROC: {ce.get('auroc', 0):.4f}",
            f"  AUPRC: {ce.get('auprc', 0):.4f}",
        ])
    
    if 'pattern_recovery' in eval_results:
        pr = eval_results['pattern_recovery']
        summary_lines.extend([
            f"",
            f"Pattern Recovery:",
            f"  Patterns: {pr.get('n_patterns', 'N/A')}",
            f"  Recovered: {pr.get('n_recovered', 'N/A')}",
            f"  Rate: {pr.get('recovery_rate', 0):.1%}",
        ])
    
    axes[1,1].text(0.1, 0.9, '\n'.join(summary_lines), fontsize=11, family='monospace',
                   verticalalignment='top', transform=axes[1,1].transAxes)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_ablation_study(ablation_results, output_path, filename="ablation_study.png"):
    print("\n  Generating ablation study plots...")
    
    if not ablation_results:
        print("    No ablation results to plot")
        return
    
    df = pd.DataFrame(ablation_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = ['auroc', 'auprc', 'recovery_rate']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(available_metrics[:3]):
            axes[0,0].bar(x + i*width, df[metric], width, label=metric)
        
        axes[0,0].set_xticks(x + width)
        axes[0,0].set_xticklabels(df['name'], rotation=45, ha='right')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Performance by Configuration')
        axes[0,0].legend()
        axes[0,0].set_ylim([0, 1])
    
    if 'sparsity' in df.columns and 'auroc' in df.columns:
        axes[0,1].scatter(df['sparsity'], df['auroc'], s=100, alpha=0.7)
        for i, row in df.iterrows():
            axes[0,1].annotate(row['name'], (row['sparsity'], row['auroc']), fontsize=8)
        axes[0,1].set_xlabel('Sparsity')
        axes[0,1].set_ylabel('AUROC')
        axes[0,1].set_title('Effect of Sparsity on Performance')
    
    if 'noise_level' in df.columns and 'auroc' in df.columns:
        axes[1,0].scatter(df['noise_level'], df['auroc'], s=100, alpha=0.7, color='coral')
        for i, row in df.iterrows():
            axes[1,0].annotate(row['name'], (row['noise_level'], row['auroc']), fontsize=8)
        axes[1,0].set_xlabel('Noise Level')
        axes[1,0].set_ylabel('AUROC')
        axes[1,0].set_title('Effect of Noise on Performance')
    
    table_cols = ['name'] + [c for c in ['n_cells', 'sparsity', 'noise_level', 'auroc', 'auprc'] if c in df.columns]
    table_data = df[table_cols].round(4).values.tolist()
    table = axes[1,1].table(cellText=table_data, colLabels=table_cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1,1].axis('off')
    axes[1,1].set_title('Configuration Summary')
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def generate_all_figures(adata, mirna_db, payload, ground_truth, gt_labels, eval_results, config: Config):
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    fig_path = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_path, exist_ok=True)
    
    if hasattr(mirna_db, 'interactions'):
        comm_patterns = CommunicationPattern(mirna_db)
        comm_matrix = comm_patterns.get_communication_matrix()
        plot_communication_matrix(comm_matrix, fig_path)
    
    plot_data_overview(adata, fig_path)
    
    plot_proxy_validation(adata, mirna_db, ground_truth, fig_path)
    
    plot_sender_receiver_scores(adata, fig_path)
    
    plot_graph_statistics(payload, fig_path)
    
    if eval_results:
        plot_evaluation_results(eval_results, fig_path)
    
    print(f"\n  All figures saved to: {fig_path}")


def run_full_pipeline(config: Config):
    
    print("\n" + "=" * 80)
    print("                    mirCCC: miRNA-mediated Cell Communication")
    print("                         Integrated Analysis Pipeline")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_output_dirs(config)
    
    results = {}
    
    print("\n" + "=" * 60)
    print("PHASE 0: Data Preparation")
    print("=" * 60)
    
    synth_path = os.path.join(config.OUTPUT_DIR, "synthetic_data")
    h5ad_path = os.path.join(synth_path, "synthetic_mirna_communication.h5ad")
    gt_path = os.path.join(synth_path, "ground_truth_labels.csv")
    gt_full_path = os.path.join(synth_path, "ground_truth_full.pkl")
    
    if os.path.exists(h5ad_path) and os.path.exists(gt_full_path):
        print(f"\n  Loading existing synthetic data from: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        gt_labels = pd.read_csv(gt_path)
        with open(gt_full_path, 'rb') as f:
            ground_truth = pickle.load(f)
        
        mirna_db = MiRNATargetDatabase(
            mir2tar_path=config.MIR2TAR_PATH,
            min_targets=config.MIN_TARGETS_PER_MIRNA
        )
    else:
        print("\n  Generating new synthetic data...")
        
        mirna_db = MiRNATargetDatabase(
            mir2tar_path=config.MIR2TAR_PATH,
            min_targets=config.MIN_TARGETS_PER_MIRNA
        )
        mirna_db.summary()
        
        generator = SyntheticDataGenerator(
            mirna_db=mirna_db,
            n_cells=config.SYNTH_N_CELLS,
            n_genes=config.SYNTH_N_GENES,
            n_cell_types=config.SYNTH_N_CELL_TYPES,
            sparsity=config.SYNTH_SPARSITY,
            noise_level=config.SYNTH_NOISE_LEVEL,
            batch_effect_strength=config.SYNTH_BATCH_EFFECT,
            random_seed=config.SYNTH_RANDOM_SEED,

            biogenesis_path=config.BIOGENESIS_PATH,
            risc_path=config.RISC_PATH,
            sorting_path=config.SORTING_PATH,
            species=getattr(config, 'SPECIES', 'Human'),
            
            prior_boost=getattr(config, 'PRIOR_BOOST', 8.0),
            
            repression_strength=getattr(config, 'REPRESSION_STRENGTH', 0.2), 
            
            mirna_coverage=getattr(config, 'MIRNA_COVERAGE', 1.0),
            protect_dropout_scale=0.1 
        )
        
        adata = generator.generate()
        gt_labels = generator.generate_ground_truth_labels()
        ground_truth = generator.get_ground_truth()
        
        os.makedirs(synth_path, exist_ok=True)
        adata.write_h5ad(h5ad_path)
        gt_labels.to_csv(gt_path, index=False)
        with open(gt_full_path, 'wb') as f:
            pickle.dump(ground_truth, f)
        
        mirna_db.interactions.to_csv(os.path.join(synth_path, "mirna_target_interactions.csv"))
        
        print(f"\n  Saved synthetic data to: {synth_path}")
    
    print(f"\n  Data loaded: {adata.n_obs} cells Ãƒâ€” {adata.n_vars} genes")
    print(f"  Ground truth: {len(ground_truth.get('communication_pairs', []))} patterns")
    print(f"  Labels: {len(gt_labels)} cell pairs")
    
    results['adata'] = adata
    results['ground_truth'] = ground_truth
    results['gt_labels'] = gt_labels
    results['mirna_db'] = mirna_db

    if hasattr(config, 'GENEINFO_PATH') and os.path.exists(config.GENEINFO_PATH):
        print("\n" + "=" * 60)
        print("PHASE 0.5: Unifying Gene Names")
        print("=" * 60)
        adata, unify_info = unify_anndata_varnames(
            adata,
            geneinfo_csv=config.GENEINFO_PATH,
            species=getattr(config, 'SPECIES', 'Human'),
            agg="sum"
        )
        results['unify_info'] = unify_info
    else:
        print("\n  [SKIP] Gene name unification: geneinfo.csv not found")
        
    adata, PR = compute_mirna_proxy(adata, config.MIR2TAR_PATH, config)
    results['proxy_matrix'] = PR
    
    adata = compute_sender_receiver_scores(adata, config)
    
    if 'gene_alias_map' in adata.uns:
        del adata.uns['gene_alias_map']
    adata_sr_path = os.path.join(synth_path, "synthetic_with_scores.h5ad")
    adata.write_h5ad(adata_sr_path)
    print(f"\n  Saved adata with scores to: {adata_sr_path}")
    
    payload = build_communication_graph(adata, config.MIR2TAR_PATH, config)
    results['payload'] = payload
    
    graph_path = os.path.join(config.OUTPUT_DIR, "results", "graph_payload.pkl.gz")
    with gzip.open(graph_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n  Saved graph to: {graph_path}")
    
    if TORCH_AVAILABLE and payload['row_col'].shape[0] > 0:
        model_path = os.path.join(config.OUTPUT_DIR, "model")
        
        model, graph, in_channels, edge_cont_dim, num_mirna = train_dgi_model(
            payload, config, model_path
        )
        results['model'] = model
        
        subgraph_list = partition_directed_graph_as_undirected(
            graph, num_parts=config.NUM_PARTS, overlap_hops=config.OVERLAP_HOPS
        )
        
        results_path = os.path.join(config.OUTPUT_DIR, "results")
        embeddings, attention, attn_path = run_inference(
            model, graph, subgraph_list, config, results_path
        )
        results['embeddings'] = embeddings
        results['attention'] = attention
        
        filtered_path = os.path.join(results_path, "filtered_attention.pkl.gz")
        filtered_edges = filter_top_attention_edges(
            attn_path, config.RETAIN_PERCENT, filtered_path
        )
        results['filtered_edges'] = filtered_edges
        
        print("\n" + "=" * 60)
        print("PHASE 5.5: Exporting Standardized Results")
        print("=" * 60)
        
        export_paths = export_communication_results(
            edges=filtered_edges,
            adata=adata,
            payload=payload,
            output_dir=results_path,
            prefix="mirage",
            formats=["csv"],
            agg_methods=["sum", "max", "topk_sum"]
        )
        results['export_paths'] = export_paths
        
        edge_df = edges_to_mirna_edge_df(filtered_edges, adata, payload)
        cellpair_df = aggregate_to_cellpair_df(edge_df, agg="max")
        results['mirna_edge_df'] = edge_df
        results['cellpair_df'] = cellpair_df
        
    else:
        if not TORCH_AVAILABLE:
            print("\n[SKIP] Training/Inference: PyTorch not available")
        else:
            print("\n[SKIP] Training/Inference: No edges in graph")
    
    eval_results = evaluate_with_ground_truth(
        adata, payload, ground_truth, gt_labels, config.MIR2TAR_PATH, config
    )
    results['evaluation'] = eval_results
    
    eval_path = os.path.join(config.OUTPUT_DIR, "results", "evaluation_results.json")
    
    def make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj
    
    with open(eval_path, 'w') as f:
        json.dump(make_serializable(eval_results), f, indent=2)
    print(f"\n  Saved evaluation results to: {eval_path}")
    
    if config.SAVE_FIGURES:
        generate_all_figures(
            adata, mirna_db, payload, ground_truth, gt_labels, eval_results, config
        )
    
    print("\n" + "=" * 80)
    print("                         PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {config.OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - Synthetic data: {synth_path}/")
    print(f"  - Graph payload: {graph_path}")
    if TORCH_AVAILABLE:
        print(f"  - Model: {model_path}/")
        print(f"  - Embeddings & Attention: {results_path}/")
    print(f"  - Evaluation: {eval_path}")
    print(f"  - Figures: {config.OUTPUT_DIR}/figures/")
    
    if 'export_paths' in results:
        print(f"\n  Standardized export files:")
        for name, path in results['export_paths'].items():
            print(f"    - {name}: {path}")
    
    return results


def run_ablation_study(config: Config, configs_to_test: List[Dict] = None):
    print("\n" + "=" * 80)
    print("                       ABLATION STUDY")
    print("=" * 80)
    
    if configs_to_test is None:
        configs_to_test = [
            {'name': 'baseline', 'SYNTH_SPARSITY': 0.85, 'SYNTH_NOISE_LEVEL': 0.3},
            {'name': 'low_sparsity', 'SYNTH_SPARSITY': 0.75, 'SYNTH_NOISE_LEVEL': 0.3},
            {'name': 'high_sparsity', 'SYNTH_SPARSITY': 0.95, 'SYNTH_NOISE_LEVEL': 0.3},
            {'name': 'low_noise', 'SYNTH_SPARSITY': 0.85, 'SYNTH_NOISE_LEVEL': 0.1},
            {'name': 'high_noise', 'SYNTH_SPARSITY': 0.85, 'SYNTH_NOISE_LEVEL': 0.5},
            {'name': 'challenging', 'SYNTH_SPARSITY': 0.92, 'SYNTH_NOISE_LEVEL': 0.5},
        ]
    
    ablation_results = []
    
    for cfg_override in configs_to_test:
        name = cfg_override.pop('name', 'unnamed')
        print(f"\n{'='*60}")
        print(f"Testing configuration: {name}")
        print(f"{'='*60}")
        
        test_config = Config()
        for k, v in cfg_override.items():
            setattr(test_config, k, v)
        test_config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, f"ablation_{name}")
        test_config.SYNTH_RANDOM_SEED = config.SYNTH_RANDOM_SEED + hash(name) % 1000
        
        try:
            results = run_full_pipeline(test_config)
            
            result_row = {
                'name': name,
                'n_cells': test_config.SYNTH_N_CELLS,
                'sparsity': test_config.SYNTH_SPARSITY,
                'noise_level': test_config.SYNTH_NOISE_LEVEL,
            }
            
            if 'evaluation' in results:
                eval_res = results['evaluation']
                if 'proxy_evaluation' in eval_res:
                    result_row['proxy_auroc'] = eval_res['proxy_evaluation'].get('mean_auroc', 0)
                    result_row['proxy_auprc'] = eval_res['proxy_evaluation'].get('mean_auprc', 0)
                if 'communication_evaluation' in eval_res:
                    result_row['auroc'] = eval_res['communication_evaluation'].get('auroc', 0)
                    result_row['auprc'] = eval_res['communication_evaluation'].get('auprc', 0)
                if 'pattern_recovery' in eval_res:
                    result_row['recovery_rate'] = eval_res['pattern_recovery'].get('recovery_rate', 0)
            
            ablation_results.append(result_row)
            
        except Exception as e:
            print(f"  Error in {name}: {e}")
            ablation_results.append({'name': name, 'error': str(e)})
        
        cfg_override['name'] = name
    
    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = os.path.join(config.OUTPUT_DIR, "ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n  Ablation results saved to: {ablation_path}")
    
    fig_path = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_path, exist_ok=True)
    plot_ablation_study(ablation_results, fig_path)
    
    return ablation_results


if __name__ == "__main__":
    
    
    config = Config()
    
    config.OUTPUT_DIR = "./mirage_synthetic_validation"
    
    config.SYNTH_N_CELLS = 3000
    config.SYNTH_N_GENES = 4000
    config.SYNTH_N_CELL_TYPES = 8
    config.SYNTH_SPARSITY = 0.45
    config.SYNTH_NOISE_LEVEL = 0.15
    config.SYNTH_RANDOM_SEED = 42

    config.TOP_S_SENDERS = 150
    config.TOP_R_RECEIVERS = 150
    config.TOP_L_PER_SENDER = 10

    config.PRIOR_BOOST = 8.0
    config.REPRESSION_STRENGTH = 0.2
    config.RETAIN_PERCENT = 1.0
    config.MIRNA_COVERAGE = 1.0
    config.PROTECT_SIGNAL = True
    
    config.NUM_EPOCHS = 500
    config.HIDDEN_DIM = 128
    config.DEVICE = "cpu"
    
    config.SAVE_FIGURES = True
    
    
    results = run_full_pipeline(config)
    
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
