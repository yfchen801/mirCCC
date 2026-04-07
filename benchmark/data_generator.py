import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import nbinom
from typing import Dict, List, Tuple, Set, Optional
import os


MIRNA_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mirna")


class RealMiRNADatabase:
    def __init__(self, 
                 mir2tar_path: str = None,
                 biogenesis_path: str = None,
                 risc_path: str = None,
                 sorting_path: str = None,
                 min_targets: int = 5,
                 max_mirnas: int = 50,
                 species: str = "Human",
                 seed: int = 42):
        
        self.rng = np.random.default_rng(seed)
        self.species = species
        
        mir2tar_path = mir2tar_path or os.path.join(MIRNA_DATA_DIR, "mir2tar.csv")
        biogenesis_path = biogenesis_path or os.path.join(MIRNA_DATA_DIR, "biogenesis.csv")
        risc_path = risc_path or os.path.join(MIRNA_DATA_DIR, "risc.csv")
        sorting_path = sorting_path or os.path.join(MIRNA_DATA_DIR, "sorting.csv")
        
        self.mir2tar, self.full_mir2tar = self._load_mir2tar_long_format(
            mir2tar_path, min_targets, max_mirnas, species
        )
        self.mirnas = list(self.mir2tar.keys())
        self.all_targets = self._get_all_targets()
        
        self.biogenesis_genes = self._load_gene_set(biogenesis_path)
        self.risc_genes = self._load_gene_set(risc_path)
        self.sorting_genes = self._load_gene_set(sorting_path)
        
        self.machinery_genes = set()
        self.machinery_genes.update(self.biogenesis_genes)
        self.machinery_genes.update(self.risc_genes)
        self.machinery_genes.update(self.sorting_genes)
        
        print(f"[RealMiRNADatabase] Loaded:")
        print(f"  miRNAs (top-{max_mirnas}): {len(self.mirnas)}")
        print(f"  miRNAs (full pool): {len(self.full_mir2tar)}")
        print(f"  Unique targets: {len(self.all_targets)}")
        print(f"  Biogenesis genes: {len(self.biogenesis_genes)}")
        print(f"  RISC genes: {len(self.risc_genes)}")
        print(f"  Sorting genes: {len(self.sorting_genes)}")
    
    def _load_mir2tar_long_format(self, path, min_targets, max_mirnas, species):
        if not os.path.exists(path):
            raise FileNotFoundError(f"mir2tar.csv not found: {path}")
        df = pd.read_csv(path)
        if 'mirna' in df.columns:
            df = df.rename(columns={'mirna': 'miRNA'})
        if 'target' in df.columns:
            df = df.rename(columns={'target': 'target_gene'})
        if 'species' in df.columns:
            df = df[df['species'].str.contains(species, case=False, na=False)]
        
        mir2tar_full = {}
        for mirna, group in df.groupby('miRNA'):
            targets = group['target_gene'].dropna().unique().tolist()
            if len(targets) >= min_targets:
                mir2tar_full[str(mirna)] = targets
        
        if len(mir2tar_full) == 0:
            for mirna, group in df.groupby('miRNA'):
                targets = group['target_gene'].dropna().unique().tolist()
                if len(targets) >= 2:
                    mir2tar_full[str(mirna)] = targets
        
        sorted_mirnas = sorted(mir2tar_full.keys(), key=lambda m: len(mir2tar_full[m]), reverse=True)
        selected = sorted_mirnas[:max_mirnas]
        mir2tar_top = {m: mir2tar_full[m] for m in selected}
        
        return mir2tar_top, mir2tar_full
    
    def _get_all_targets(self):
        all_t = set()
        for targets in self.mir2tar.values():
            all_t.update(targets)
        for targets in self.full_mir2tar.values():
            all_t.update(targets)
        return list(all_t)
    
    def _load_gene_set(self, path):
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
        for col in ['gene', 'Gene', 'symbol', 'Symbol', 'gene_name']:
            if col in df.columns:
                genes = df[col].dropna().astype(str).str.strip().tolist()
                return [g for g in genes if g]
        return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    
    def get_targets(self, mirna):
        if mirna in self.mir2tar:
            return self.mir2tar[mirna]
        return self.full_mir2tar.get(mirna, [])


class BenchmarkDataGenerator:
    
    def __init__(self, 
                 n_cells=2000, n_genes=2000, n_cell_types=6, n_patterns=8,
                 dropout=0.15, noise=0.2, effect=0.7, seed=42,
                 n_active_per_pattern=40, n_decoys=100, signal_boost=8.0,
                 max_mirnas=50, min_targets=5, baseline_mode=True,
                 signal_protection=0.5, machinery_level=200.0,
                 background_level=20.0, target_background=100.0,
                 target_repressed=1.0,
                 true_mirnas_per_pattern=8,
                 easy_negative_ratio=0.6,
                 n_antifallback_cells=15,
                 cis_repress_factor=0.15,
                 key_mirna_fraction=0.35,
                 key_repression_multiplier=0.1,
                 support_repression_multiplier=4.0,
                 gradient_strength=0.6,
                 overlap_penalty=True,
                 ortho_min_targets=20,
                 ortho_max_targets=500,
                 ):
        
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_patterns = n_patterns
        self.dropout = dropout
        self.noise = noise
        self.effect = effect
        self.seed = seed
        self.n_active_per_pattern = n_active_per_pattern
        self.n_decoys = n_decoys if not baseline_mode else 0
        self.signal_boost = signal_boost
        self.baseline_mode = baseline_mode
        self.signal_protection = signal_protection
        self.machinery_level = machinery_level
        self.background_level = background_level
        self.target_background = target_background
        self.target_repressed = target_repressed
        
        self.true_mirnas_per_pattern = true_mirnas_per_pattern
        self.easy_negative_ratio = easy_negative_ratio
        self.n_antifallback_cells = n_antifallback_cells
        self.cis_repress_factor = cis_repress_factor
        
        self.key_mirna_fraction = key_mirna_fraction
        self.key_repression_multiplier = key_repression_multiplier
        self.support_repression_multiplier = support_repression_multiplier
        self.gradient_strength = gradient_strength
        self.overlap_penalty = overlap_penalty
        
        self.ortho_min_targets = ortho_min_targets
        self.ortho_max_targets = ortho_max_targets
        
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        
        self.cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte',
                          'Macrophage', 'DC', 'Neutrophil', 'Tumor'][:n_cell_types]
        
        self.mirna_db = RealMiRNADatabase(
            max_mirnas=max_mirnas, min_targets=min_targets, seed=seed
        )
        
        self.patterns = self._define_patterns()
        self.ground_truth = {
            'communication_pairs': [],
            'mirna_activity': {},
            'cell_assignments': None,
            'true_sender_cells': set(),
            'true_receiver_cells': set(),
            'decoy_sender_cells': set(),
            'decoy_receiver_cells': set(),
            'decoy_weak_receivers': set(),
        }
        
        self._repression_records = []
    
    def _compute_target_overlap(self, mirna_group_a: List[str], mirna_group_b: List[str]) -> float:
        targets_a = set()
        for m in mirna_group_a:
            targets_a.update(self.mirna_db.get_targets(m))
        targets_b = set()
        for m in mirna_group_b:
            targets_b.update(self.mirna_db.get_targets(m))
        
        if not targets_a or not targets_b:
            return 0.0
        intersection = len(targets_a & targets_b)
        union = len(targets_a | targets_b)
        return intersection / union if union > 0 else 0.0
    
    def _select_orthogonal_mirnas(self, n_groups: int, mirnas_per_group: int) -> List[List[str]]:
        mir2tar_full = self.mirna_db.full_mir2tar
        
        candidates = {}
        for m, targets in mir2tar_full.items():
            n_t = len(targets)
            if self.ortho_min_targets <= n_t <= self.ortho_max_targets:
                candidates[m] = set(targets)
        
        print(f"  [orthogonal] Candidate pool: {len(candidates)} miRNAs "
              f"(targets in [{self.ortho_min_targets}, {self.ortho_max_targets}])")
        
        total_needed = n_groups * mirnas_per_group
        if len(candidates) < total_needed:
            candidates = {}
            for m, targets in mir2tar_full.items():
                if len(targets) >= self.ortho_min_targets:
                    candidates[m] = set(targets)
            print(f"  [orthogonal] Relaxed max_targets -> {len(candidates)} candidates")
        
        if len(candidates) < total_needed:
            candidates = {m: set(t) for m, t in mir2tar_full.items() if len(t) >= 5}
            print(f"  [orthogonal] Relaxed to min=5 -> {len(candidates)} candidates")
        
        groups = []
        used_mirnas = set()
        used_targets_by_group = []
        all_used_targets = set()
        
        for g in range(n_groups):
            available = {m: t for m, t in candidates.items() if m not in used_mirnas}
            
            if len(available) < mirnas_per_group:
                print(f"  [orthogonal] Group {g}: only {len(available)} available, using all")
                group = list(available.keys())
                groups.append(group)
                used_mirnas.update(group)
                gt = set()
                for m in group:
                    gt.update(candidates.get(m, set()))
                used_targets_by_group.append(gt)
                all_used_targets.update(gt)
                continue
            
            group = []
            group_targets = set()
            selected_in_group = set()
            
            for _ in range(mirnas_per_group):
                best_m = None
                best_score = -1
                
                for m, t in available.items():
                    if m in selected_in_group:
                        continue
                    
                    if all_used_targets:
                        cross_overlap = len(t & all_used_targets) / len(t)
                    else:
                        cross_overlap = 0.0
                    
                    if group_targets:
                        intra_overlap = len(t & group_targets) / len(t)
                    else:
                        intra_overlap = 0.0
                    
                    score = (1.0 - cross_overlap) * 0.7 + (1.0 - intra_overlap) * 0.3
                    
                    target_bonus = min(len(t) / 100.0, 1.0) * 0.05
                    score += target_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_m = m
                
                if best_m is None:
                    break
                
                group.append(best_m)
                selected_in_group.add(best_m)
                group_targets.update(candidates[best_m])
            
            groups.append(group)
            used_mirnas.update(group)
            used_targets_by_group.append(group_targets)
            all_used_targets.update(group_targets)
            
            print(f"  [orthogonal] Group {g}: {len(group)} miRNAs, "
                  f"{len(group_targets)} unique targets")
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ti = used_targets_by_group[i]
                tj = used_targets_by_group[j]
                if ti and tj:
                    jaccard = len(ti & tj) / len(ti | tj)
                    overlap_i = len(ti & tj) / len(ti)
                    overlap_j = len(ti & tj) / len(tj)
                    print(f"  [orthogonal] Group {i} vs {j}: "
                          f"Jaccard={jaccard:.3f}, "
                          f"|i∩j|/|i|={overlap_i:.3f}, "
                          f"|i∩j|/|j|={overlap_j:.3f}")
        
        for group in groups:
            for m in group:
                if m not in self.mirna_db.mir2tar:
                    self.mirna_db.mir2tar[m] = self.mirna_db.full_mir2tar[m]
                if m not in self.mirna_db.mirnas:
                    self.mirna_db.mirnas.append(m)
        
        return groups
    
    def _define_patterns(self):
        templates = [
            ('Tumor', 'T_cell', 0.85), ('Tumor', 'Macrophage', 0.8),
            ('Macrophage', 'Tumor', 0.7), ('T_cell', 'B_cell', 0.65),
            ('Monocyte', 'Macrophage', 0.6), ('NK_cell', 'Tumor', 0.55),
            ('DC', 'T_cell', 0.5), ('Neutrophil', 'Tumor', 0.45),
        ]
        
        valid_templates = [(s, r, st) for s, r, st in templates[:self.n_patterns] 
                          if s in self.cell_types and r in self.cell_types]
        n_actual = len(valid_templates)
        
        if self.baseline_mode:
            mir_per = max(2, len(self.mirna_db.mirnas) // max(1, n_actual))
        else:
            mir_per = self.true_mirnas_per_pattern
        
        if self.overlap_penalty and not self.baseline_mode and n_actual >= 2:
            print(f"\n[v14: Orthogonal miRNA selection for {n_actual} patterns, {mir_per} miRNAs each]")
            mirna_groups = self._select_orthogonal_mirnas(n_actual, mir_per)
        else:
            mirna_groups = []
            idx = 0
            for _ in range(n_actual):
                mirs = self.mirna_db.mirnas[idx:idx + mir_per]
                mirna_groups.append(list(mirs))
                idx += mir_per
        
        patterns = []
        for i, (s, r, strength) in enumerate(valid_templates):
            if i < len(mirna_groups) and mirna_groups[i]:
                patterns.append({
                    'sender': s, 'receiver': r, 'mirnas': mirna_groups[i],
                    'effect_strength': strength * self.effect
                })
        
        self._true_mirna_end_idx = sum(len(g) for g in mirna_groups)
        return patterns
    
    def _build_genes(self):
        genes = []
        machinery = set()
        machinery.update(self.mirna_db.biogenesis_genes)
        machinery.update(self.mirna_db.risc_genes)
        machinery.update(self.mirna_db.sorting_genes)
        genes.extend(list(machinery))
        remaining_slots = self.n_genes - len(genes)
        targets = [t for t in self.mirna_db.all_targets if t not in machinery]
        if len(targets) > remaining_slots:
            selected_targets = list(self.rng.choice(targets, remaining_slots, replace=False))
        else:
            selected_targets = targets
        genes.extend(selected_targets)
        genes_set = set(genes)
        i = 0
        while len(genes) < self.n_genes:
            placeholder = f"GENE{i:05d}"
            if placeholder not in genes_set:
                genes.append(placeholder)
                genes_set.add(placeholder)
            i += 1
        return genes[:self.n_genes]
    
    def _get_mirna_group_targets(self, mirnas):
        targets = set()
        for m in mirnas:
            targets.update(self.mirna_target_idx.get(m, []))
        return sorted(targets)
    
    def generate(self):
        has_traps = (self.overlap_penalty or 
                     self.key_mirna_fraction < 0.99 or 
                     self.gradient_strength > 0.01)
        if self.baseline_mode:
            mode_str = "BASELINE (legacy)"
        elif has_traps:
            mode_str = "TRAP v14 (orthogonal+variable+gradient)"
        else:
            mode_str = "BASE (trap structure, traps disabled)"
        
        print(f"\n{'='*60}")
        print(f"Generating Benchmark Data - {mode_str}")
        print(f"{'='*60}")
        print(f"  Cells: {self.n_cells}, Genes: {self.n_genes}")
        print(f"  Active per pattern: {self.n_active_per_pattern}")
        if not self.baseline_mode:
            print(f"  True miRNAs/pattern: {self.true_mirnas_per_pattern}")
            print(f"  Key miRNA fraction: {self.key_mirna_fraction}")
            print(f"  Key repression mult: {self.key_repression_multiplier}")
            print(f"  Gradient strength: {self.gradient_strength}")
            if self.overlap_penalty:
                print(f"  Ortho target range: [{self.ortho_min_targets}, {self.ortho_max_targets}]")
            print(f"  Easy negative ratio: {self.easy_negative_ratio}")
        
        self.genes = self._build_genes()
        self.g2i = {g: i for i, g in enumerate(self.genes)}
        
        self.biog_idx = [self.g2i[g] for g in self.mirna_db.biogenesis_genes if g in self.g2i]
        self.risc_idx = [self.g2i[g] for g in self.mirna_db.risc_genes if g in self.g2i]
        self.sort_idx = [self.g2i[g] for g in self.mirna_db.sorting_genes if g in self.g2i]
        self.machinery_idx = set(self.biog_idx + self.risc_idx + self.sort_idx)
        
        self.mirna_target_idx = {}
        for mir in self.mirna_db.mirnas:
            targets = self.mirna_db.get_targets(mir)
            idx = [self.g2i[t] for t in targets if t in self.g2i and self.g2i[t] not in self.machinery_idx]
            if idx:
                self.mirna_target_idx[mir] = idx
        
        all_target_idx = set()
        for idx_list in self.mirna_target_idx.values():
            all_target_idx.update(idx_list)
        self.all_target_idx = list(all_target_idx)
        
        print(f"  Per-miRNA target mapping: {len(self.mirna_target_idx)} miRNAs with targets in genes")
        
        expr = self._gen_base_expr()
        expr = self._add_markers(expr)
        all_active_senders, all_active_receivers = self._collect_active_cells()
        expr = self._set_background(expr)
        expr = self._embed_true_communication(expr, all_active_senders, all_active_receivers)
        
        if not self.baseline_mode:
            print(f"\n[v14: No decoy cells - hard negatives from cross-pattern pairs]")
            self.ground_truth['pattern_decoy_map'] = {}
        
        self._verify_signals(expr, all_active_senders, all_active_receivers)
        expr = self._add_noise_protected(expr, all_active_senders, all_active_receivers)
        expr = self._apply_dropout(expr)
        self._verify_signals(expr, all_active_senders, all_active_receivers, stage="after noise/dropout")
        
        adata = self._create_adata(expr)
        gt_labels = self._create_gt_labels()
        self.ground_truth['communication_matrix'] = self._create_comm_matrix()
        
        print(f"\n[Statistics]")
        print(f"  True senders: {len(self.ground_truth['true_sender_cells'])}")
        print(f"  True receivers: {len(self.ground_truth['true_receiver_cells'])}")
        print(f"  GT Labels: {len(gt_labels)}")
        print(f"  Data shape: {adata.shape}")
        gene_sets = adata.uns
        print(f"  Gene sets: bio={len(gene_sets.get('biogenesis_genes', []))}, "
              f"risc={len(gene_sets.get('risc_genes', []))}, "
              f"sort={len(gene_sets.get('sorting_genes', []))}")
        print(f"  miRNA targets: {len(gene_sets.get('mirna_list', []))} miRNAs")
        if 'pair_type' in gt_labels.columns:
            print(f"  Pair types: {gt_labels['pair_type'].value_counts().to_dict()}")
        
        return adata, self.ground_truth, gt_labels
    
    def _gen_base_expr(self):
        n_per = self.n_cells // len(self.cell_types)
        expr = np.zeros((self.n_cells, len(self.genes)))
        self.cell_labels = []
        idx = 0
        for ci, ct in enumerate(self.cell_types):
            n_ct = n_per if ci < len(self.cell_types) - 1 else self.n_cells - idx
            tf = self.rng.gamma(2, 0.5, len(self.genes))
            for _ in range(n_ct):
                means = 2.5 * tf * self.rng.gamma(2, 0.5)
                r, p = 2.0, 2.0 / (2.0 + means + 1e-10)
                expr[idx] = nbinom.rvs(r, np.clip(p, 0.01, 0.99))
                self.cell_labels.append(ct)
                idx += 1
        self.ground_truth['cell_assignments'] = self.cell_labels.copy()
        return expr
    
    def _add_markers(self, expr):
        for ci, ct in enumerate(self.cell_types):
            cells = [i for i, c in enumerate(self.cell_labels) if c == ct]
            if not cells:
                continue
            n_markers = min(20, len(self.genes) // self.n_cell_types)
            start = ci * n_markers
            marker_idx = list(range(start, min(start + n_markers, len(self.genes))))
            if marker_idx:
                boost = self.rng.uniform(3, 8, len(marker_idx))
                for cell_i in cells:
                    expr[cell_i, marker_idx] *= boost
        return expr
    
    def _collect_active_cells(self):
        print("\n[Collecting active cells]")
        all_active_senders = set()
        all_active_receivers = set()
        for pat in self.patterns:
            s_cells = [i for i, c in enumerate(self.cell_labels) if c == pat['sender']]
            r_cells = [i for i, c in enumerate(self.cell_labels) if c == pat['receiver']]
            if not s_cells or not r_cells:
                continue
            n_s = min(self.n_active_per_pattern, len(s_cells))
            n_r = min(self.n_active_per_pattern, len(r_cells))
            active_s = list(self.rng.choice(s_cells, n_s, replace=False))
            active_r = list(self.rng.choice(r_cells, n_r, replace=False))
            all_active_senders.update(active_s)
            all_active_receivers.update(active_r)
            self.ground_truth['communication_pairs'].append({
                'sender': pat['sender'], 'receiver': pat['receiver'],
                'mirnas': pat['mirnas'], 'strength': pat['effect_strength'],
                'sender_cells': list(active_s), 'receiver_cells': list(active_r),
                'is_true': True
            })
        all_active_senders = list(all_active_senders)
        all_active_receivers = list(all_active_receivers)
        self.ground_truth['true_sender_cells'] = set(all_active_senders)
        self.ground_truth['true_receiver_cells'] = set(all_active_receivers)
        print(f"  Total active senders: {len(all_active_senders)}")
        print(f"  Total active receivers: {len(all_active_receivers)}")
        return all_active_senders, all_active_receivers
    
    def _set_background(self, expr):
        print("\n[Setting background expression]")
        base_bg = self.target_background * 0.5
        non_machinery = [i for i in range(len(self.genes)) if i not in self.machinery_idx]
        for cell in range(self.n_cells):
            variation = self.rng.uniform(0.5, 1.5, len(non_machinery))
            expr[cell, non_machinery] = np.maximum(expr[cell, non_machinery], base_bg * variation)
        for cell in range(self.n_cells):
            if self.biog_idx:
                expr[cell, self.biog_idx] = self.background_level * self.rng.uniform(0.5, 1.5, len(self.biog_idx))
            if self.sort_idx:
                expr[cell, self.sort_idx] = self.background_level * self.rng.uniform(0.5, 1.5, len(self.sort_idx))
            if self.risc_idx:
                expr[cell, self.risc_idx] = self.background_level * self.rng.uniform(0.5, 1.5, len(self.risc_idx))
        print(f"  Non-machinery background: ~{base_bg:.1f}")
        print(f"  Machinery background: ~{self.background_level:.1f}")
        return expr
    
    def _embed_true_communication(self, expr, all_active_senders, all_active_receivers):
        print(f"\n[Embedding TRUE communication - v14 orthogonal+variable+gradient]")
        
        if self.biog_idx and all_active_senders:
            expr[np.ix_(all_active_senders, self.biog_idx)] = self.machinery_level
        if self.sort_idx and all_active_senders:
            expr[np.ix_(all_active_senders, self.sort_idx)] = self.machinery_level
        if self.risc_idx and all_active_receivers:
            expr[np.ix_(all_active_receivers, self.risc_idx)] = self.machinery_level
        
        for pair_info in self.ground_truth['communication_pairs']:
            if not pair_info.get('is_true', True):
                continue
            mirnas = pair_info['mirnas']
            sender_cells = pair_info['sender_cells']
            receiver_cells = pair_info['receiver_cells']
            
            if not mirnas:
                continue
            
            n_key = max(1, int(len(mirnas) * self.key_mirna_fraction))
            mir_shuffled = list(mirnas)
            self.rng.shuffle(mir_shuffled)
            key_mirnas = mir_shuffled[:n_key]
            support_mirnas = mir_shuffled[n_key:]
            
            pair_info['key_mirnas'] = list(key_mirnas)
            pair_info['support_mirnas'] = list(support_mirnas)
            
            print(f"    {pair_info['sender']}->{pair_info['receiver']}: "
                  f"{len(key_mirnas)} key + {len(support_mirnas)} support miRNAs")
            
            for rank, m in enumerate(key_mirnas):
                tidx = self.mirna_target_idx.get(m, [])
                if not tidx:
                    continue
                pt_arr = np.array(tidx)
                
                gradient_factor = 1.0 + rank * self.gradient_strength / max(1, n_key - 1)
                
                recv_level = self.target_repressed * self.key_repression_multiplier * gradient_factor
                for cell in receiver_cells:
                    v = self.rng.uniform(0.9, 1.1, len(pt_arr))
                    expr[cell, pt_arr] = recv_level * v
                    self._repression_records.append((cell, pt_arr, recv_level))
                
                sender_level = recv_level * 3.0
                for cell in sender_cells:
                    v = self.rng.uniform(0.8, 1.2, len(pt_arr))
                    expr[cell, pt_arr] = sender_level * v
                    self._repression_records.append((cell, pt_arr, sender_level))
            
            for rank, m in enumerate(support_mirnas):
                tidx = self.mirna_target_idx.get(m, [])
                if not tidx:
                    continue
                pt_arr = np.array(tidx)
                
                gradient_factor = 1.0 + rank * self.gradient_strength / max(1, len(support_mirnas) - 1)
                
                recv_level = self.target_repressed * self.support_repression_multiplier * gradient_factor
                for cell in receiver_cells:
                    v = self.rng.uniform(0.85, 1.15, len(pt_arr))
                    expr[cell, pt_arr] = recv_level * v
                    self._repression_records.append((cell, pt_arr, recv_level))
                
                sender_level = recv_level * 2.0
                for cell in sender_cells:
                    v = self.rng.uniform(0.8, 1.2, len(pt_arr))
                    expr[cell, pt_arr] = sender_level * v
                    self._repression_records.append((cell, pt_arr, sender_level))
            
            all_targets = self._get_mirna_group_targets(mirnas)
            key_targets = self._get_mirna_group_targets(key_mirnas)
            if all_targets and receiver_cells:
                recv_mean = expr[np.ix_(receiver_cells, all_targets[:30])].mean()
                key_recv = expr[np.ix_(receiver_cells, key_targets[:20])].mean() if key_targets else 0
                print(f"      Recv target mean: all={recv_mean:.1f}, key={key_recv:.1f}, bg={self.target_background:.1f}")
        
        print(f"  Senders: biogenesis={self.machinery_level}, sorting={self.machinery_level}")
        print(f"  Receivers: RISC={self.machinery_level}")
        return expr
    
    def _verify_signals(self, expr, all_active_senders, all_active_receivers, stage="initial"):
        print(f"\n[Verifying signals - {stage}]")
        active_cells = set(all_active_senders) | set(all_active_receivers)
        other_cells = [i for i in range(self.n_cells) if i not in active_cells]
        if self.biog_idx and all_active_senders:
            sender_biog = expr[np.ix_(all_active_senders, self.biog_idx)].mean()
            other_biog = expr[np.ix_(other_cells, self.biog_idx)].mean() if other_cells else 0
            ratio = sender_biog / other_biog if other_biog > 0 else 0
            print(f"  Biogenesis: senders={sender_biog:.1f}, others={other_biog:.1f}, ratio={ratio:.1f}x")
        if self.risc_idx and all_active_receivers:
            recv_risc = expr[np.ix_(all_active_receivers, self.risc_idx)].mean()
            other_risc = expr[np.ix_(other_cells, self.risc_idx)].mean() if other_cells else 0
            ratio = recv_risc / other_risc if other_risc > 0 else 0
            print(f"  RISC: receivers={recv_risc:.1f}, others={other_risc:.1f}, ratio={ratio:.1f}x")
        if self.ground_truth['communication_pairs']:
            p0 = self.ground_truth['communication_pairs'][0]
            pt_idx = self._get_mirna_group_targets(p0['mirnas'])
            key_idx = self._get_mirna_group_targets(p0.get('key_mirnas', []))
            if pt_idx and p0['receiver_cells']:
                recv_target = expr[np.ix_(p0['receiver_cells'], pt_idx[:30])].mean()
                other_target = expr[np.ix_(other_cells[:100], pt_idx[:30])].mean() if other_cells else 0
                print(f"  Pattern0 targets: receivers={recv_target:.1f}, others={other_target:.1f}")
                if key_idx:
                    key_recv = expr[np.ix_(p0['receiver_cells'], key_idx[:15])].mean()
                    print(f"  Pattern0 KEY targets: receivers={key_recv:.1f}")
    
    def _add_noise_protected(self, expr, all_active_senders, all_active_receivers):
        decoy_senders = list(self.ground_truth.get('decoy_sender_cells', set()))
        decoy_receivers = list(self.ground_truth.get('decoy_receiver_cells', set()))
        
        all_senders = list(set(all_active_senders) | set(decoy_senders))
        all_receivers = list(set(all_active_receivers) | set(decoy_receivers))
        
        noise = self.rng.lognormal(0, self.noise, expr.shape)
        expr = expr * noise
        
        if self.signal_protection > 0:
            if self.biog_idx and all_senders:
                noisy = expr[np.ix_(all_senders, self.biog_idx)]
                expr[np.ix_(all_senders, self.biog_idx)] = (
                    noisy * (1 - self.signal_protection) + 
                    self.machinery_level * self.signal_protection
                )
            if self.sort_idx and all_senders:
                noisy = expr[np.ix_(all_senders, self.sort_idx)]
                expr[np.ix_(all_senders, self.sort_idx)] = (
                    noisy * (1 - self.signal_protection) + 
                    self.machinery_level * self.signal_protection
                )
            if self.risc_idx and all_receivers:
                noisy = expr[np.ix_(all_receivers, self.risc_idx)]
                expr[np.ix_(all_receivers, self.risc_idx)] = (
                    noisy * (1 - self.signal_protection) + 
                    self.machinery_level * self.signal_protection
                )
            
            target_protection = self.signal_protection * 0.8
            for cell, target_arr, target_val in self._repression_records:
                noisy = expr[cell, target_arr]
                expr[cell, target_arr] = (
                    noisy * (1 - target_protection) + 
                    target_val * target_protection
                )
        
        return expr
    
    def _apply_dropout(self, expr):
        signal_genes = set(self.all_target_idx) if self.all_target_idx else set()
        signal_genes.update(self.machinery_idx)
        mask = self.rng.random(expr.shape) < self.dropout
        effective_dropout = self.dropout * (1 - self.signal_protection)
        for idx in signal_genes:
            if idx < self.n_genes:
                mask[:, idx] = self.rng.random(self.n_cells) < effective_dropout
        
        all_senders = list(set(self.ground_truth['true_sender_cells']) | 
                          self.ground_truth.get('decoy_sender_cells', set()))
        all_receivers = list(set(self.ground_truth['true_receiver_cells']) | 
                           self.ground_truth.get('decoy_receiver_cells', set()))
        machinery_dropout = self.dropout * (1 - self.signal_protection) ** 2
        for idx in self.biog_idx + self.sort_idx:
            if all_senders:
                mask[all_senders, idx] = self.rng.random(len(all_senders)) < machinery_dropout
        for idx in self.risc_idx:
            if all_receivers:
                mask[all_receivers, idx] = self.rng.random(len(all_receivers)) < machinery_dropout
        expr[mask] = 0
        return expr
    
    def _create_adata(self, expr):
        X = sparse.csr_matrix(expr)
        obs = pd.DataFrame({
            'cell_type': self.cell_labels, 
            'cell_id': [f"cell_{i}" for i in range(self.n_cells)]
        })
        obs.index = obs['cell_id']
        var = pd.DataFrame({'gene_name': self.genes})
        var.index = var['gene_name']
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.uns['mirna_targets'] = {m: self.mirna_db.get_targets(m) for m in self.mirna_db.mirnas}
        adata.uns['mirna_list'] = self.mirna_db.mirnas
        adata.uns['biogenesis_genes'] = self.mirna_db.biogenesis_genes
        adata.uns['risc_genes'] = self.mirna_db.risc_genes
        adata.uns['sorting_genes'] = self.mirna_db.sorting_genes
        adata.uns['true_sender_cells'] = list(self.ground_truth['true_sender_cells'])
        adata.uns['true_receiver_cells'] = list(self.ground_truth['true_receiver_cells'])
        adata.uns['decoy_sender_cells'] = list(self.ground_truth['decoy_sender_cells'])
        adata.uns['decoy_receiver_cells'] = list(self.ground_truth['decoy_receiver_cells'])
        adata.uns['decoy_weak_receivers'] = list(self.ground_truth.get('decoy_weak_receivers', set()))
        return adata
    
    def _create_gt_labels(self):
        records = []
        print(f"\n[GT Labels - v14 (orthogonal cross-pattern)]")
        
        true_patterns = [p for p in self.ground_truth['communication_pairs'] if p.get('is_true')]
        
        for pat in true_patterns:
            sender_cells, receiver_cells = pat['sender_cells'], pat['receiver_cells']
            n_pairs = min(80, len(sender_cells) * len(receiver_cells))
            
            sampled = set()
            attempts = 0
            while len(sampled) < n_pairs and attempts < n_pairs * 10:
                s, r = int(self.rng.choice(sender_cells)), int(self.rng.choice(receiver_cells))
                if s != r and (s, r) not in sampled:
                    sampled.add((s, r))
                    records.append({
                        'sender_cell': s, 'receiver_cell': r, 'communication': 1,
                        'sender_type': self.cell_labels[s], 
                        'receiver_type': self.cell_labels[r],
                        'pair_type': 'true_positive'
                    })
                attempts += 1
        
        n_pos = len(records)
        sampled_all = set((r['sender_cell'], r['receiver_cell']) for r in records)
        print(f"  True pairs (positive): {n_pos}")
        
        n_total_neg = n_pos
        n_easy_neg = int(n_total_neg * self.easy_negative_ratio)
        n_hard_neg = n_total_neg - n_easy_neg
        
        hard_count = 0
        if not self.baseline_mode and len(true_patterns) >= 2:
            for i, pi in enumerate(true_patterns):
                ti = set(self._get_mirna_group_targets(pi['mirnas']))
                for j, pj in enumerate(true_patterns):
                    if i >= j:
                        continue
                    tj = set(self._get_mirna_group_targets(pj['mirnas']))
                    overlap = len(ti & tj) / max(len(ti | tj), 1)
                    print(f"    Target overlap P{i}({pi['sender']}->{pi['receiver']}) vs "
                          f"P{j}({pj['sender']}->{pj['receiver']}): {overlap:.1%}")
            
            cross_pairs_pool = []
            for i, pat_i in enumerate(true_patterns):
                for j, pat_j in enumerate(true_patterns):
                    if i == j:
                        continue
                    for s in pat_i['sender_cells']:
                        for r in pat_j['receiver_cells']:
                            if s != r and (s, r) not in sampled_all:
                                cross_pairs_pool.append((int(s), int(r), 
                                    self.cell_labels[s], self.cell_labels[r]))
            
            self.rng.shuffle(cross_pairs_pool)
            
            for s, r, st, rt in cross_pairs_pool[:n_hard_neg]:
                if (s, r) not in sampled_all:
                    sampled_all.add((s, r))
                    records.append({
                        'sender_cell': s, 'receiver_cell': r, 'communication': 0,
                        'sender_type': st, 'receiver_type': rt,
                        'pair_type': 'hard_negative'
                    })
                    hard_count += 1
        
        print(f"  Hard negatives (cross-pattern): {hard_count}")
        
        easy_count = 0
        half_signal_count = 0
        
        all_used = self.ground_truth['true_sender_cells'] | self.ground_truth['true_receiver_cells']
        all_unused = [i for i in range(self.n_cells) if i not in all_used]
        self.rng.shuffle(all_unused)
        
        if self.baseline_mode:
            n_half_signal = int(n_total_neg * 0.8)
            n_sender_only = n_half_signal // 2
            n_receiver_only = n_half_signal - n_sender_only
            n_easy_needed = n_total_neg - n_half_signal
            
            active_senders = list(self.ground_truth['true_sender_cells'])
            active_receivers = list(self.ground_truth['true_receiver_cells'])
            
            non_receiver_pool = [i for i in range(self.n_cells) 
                                 if i not in self.ground_truth['true_receiver_cells']]
            non_sender_pool = [i for i in range(self.n_cells) 
                              if i not in self.ground_truth['true_sender_cells']]
            
            attempts, pairs_created = 0, 0
            while pairs_created < n_sender_only and attempts < n_sender_only * 50:
                s = int(self.rng.choice(active_senders))
                r = int(self.rng.choice(non_receiver_pool))
                if s != r and (s, r) not in sampled_all:
                    sampled_all.add((s, r))
                    records.append({
                        'sender_cell': s, 'receiver_cell': r, 'communication': 0,
                        'sender_type': self.cell_labels[s],
                        'receiver_type': self.cell_labels[r],
                        'pair_type': 'half_neg_sender_only'
                    })
                    pairs_created += 1
                    half_signal_count += 1
                attempts += 1
            
            attempts, pairs_created = 0, 0
            while pairs_created < n_receiver_only and attempts < n_receiver_only * 50:
                s = int(self.rng.choice(non_sender_pool))
                r = int(self.rng.choice(active_receivers))
                if s != r and (s, r) not in sampled_all:
                    sampled_all.add((s, r))
                    records.append({
                        'sender_cell': s, 'receiver_cell': r, 'communication': 0,
                        'sender_type': self.cell_labels[s],
                        'receiver_type': self.cell_labels[r],
                        'pair_type': 'half_neg_receiver_only'
                    })
                    pairs_created += 1
                    half_signal_count += 1
                attempts += 1
            
            print(f"  Half-signal negatives: {half_signal_count} "
                  f"(sender_only + receiver_only)")
            
            if len(all_unused) >= 4:
                half = len(all_unused) // 2
                rand_sender_pool = all_unused[:half]
                rand_receiver_pool = all_unused[half:]
                
                attempts, pairs_created = 0, 0
                while pairs_created < n_easy_needed and attempts < n_easy_needed * 50:
                    s = int(self.rng.choice(rand_sender_pool))
                    r = int(self.rng.choice(rand_receiver_pool))
                    if s != r and (s, r) not in sampled_all:
                        sampled_all.add((s, r))
                        records.append({
                            'sender_cell': s, 'receiver_cell': r, 'communication': 0,
                            'sender_type': self.cell_labels[s],
                            'receiver_type': self.cell_labels[r],
                            'pair_type': 'easy_negative'
                        })
                        pairs_created += 1
                        easy_count += 1
                    attempts += 1
            
            print(f"  Easy negatives (random): {easy_count}")
        else:
            n_easy_needed = n_easy_neg
            
            if len(all_unused) >= 4:
                half = len(all_unused) // 2
                rand_sender_pool = all_unused[:half]
                rand_receiver_pool = all_unused[half:]
                
                attempts, pairs_created = 0, 0
                while pairs_created < n_easy_needed and attempts < n_easy_needed * 50:
                    s = int(self.rng.choice(rand_sender_pool))
                    r = int(self.rng.choice(rand_receiver_pool))
                    if s != r and (s, r) not in sampled_all:
                        sampled_all.add((s, r))
                        records.append({
                            'sender_cell': s, 'receiver_cell': r, 'communication': 0,
                            'sender_type': self.cell_labels[s],
                            'receiver_type': self.cell_labels[r],
                            'pair_type': 'easy_negative'
                        })
                        pairs_created += 1
                        easy_count += 1
                    attempts += 1
                
                print(f"  Easy negatives (random): {easy_count}")
        
        actual_easy_ratio = easy_count / max(1, easy_count + hard_count + half_signal_count)
        total_neg = hard_count + half_signal_count + easy_count
        print(f"  Total: {n_pos} pos + {hard_count} hard_neg + {half_signal_count} half_neg + {easy_count} easy_neg = {len(records)}")
        print(f"  Easy ratio: {actual_easy_ratio:.2f} (target: {self.easy_negative_ratio:.2f})")
        
        has_traps = (self.overlap_penalty or 
                     self.key_mirna_fraction < 0.99 or 
                     self.gradient_strength > 0.01)
        
        if has_traps:
            print(f"\n  === v14 Trap Design ===")
            print(f"  1. Orthogonal: {'ON' if self.overlap_penalty else 'OFF'}")
            print(f"  2. Variable importance: {self.key_mirna_fraction:.0%} key miRNAs with {self.key_repression_multiplier}x repression")
            print(f"  3. Gradient: strength={self.gradient_strength}")
            print(f"  -> Baselines: MEAN*MEAN -> cannot distinguish true from cross")
            print(f"  -> miRAGE: per-miRNA matching + attention -> can distinguish")
        elif not self.baseline_mode:
            print(f"\n  === Base Design (trap structure, traps disabled) ===")
            print(f"  Uniform repression, no orthogonal selection, no gradient")
            print(f"  {hard_count} cross-pattern negatives + {easy_count} easy negatives")
            print(f"  -> All methods should achieve ~0.95 AUROC")
        else:
            print(f"\n  === Legacy Baseline Design ===")
        
        return pd.DataFrame(records)
    
    def _create_comm_matrix(self):
        mat = pd.DataFrame(0.0, index=self.cell_types, columns=self.cell_types)
        for pat in self.patterns:
            mat.loc[pat['sender'], pat['receiver']] = pat['effect_strength']
        return mat


def generate_baseline_standard(seed=0):
    gen = BenchmarkDataGenerator(
        n_cells=2000, n_genes=2000, n_cell_types=8, n_patterns=3,
        dropout=0.15, noise=0.2, effect=0.7,
        n_active_per_pattern=20, n_decoys=0,
        max_mirnas=50, min_targets=5,
        baseline_mode=False,
        signal_protection=0.3, machinery_level=150.0,
        background_level=40.0, target_background=80.0, target_repressed=5.0,
        true_mirnas_per_pattern=16,
        easy_negative_ratio=0.90,
        n_antifallback_cells=0,
        cis_repress_factor=0.15,
        key_mirna_fraction=1.0,
        key_repression_multiplier=1.0,
        support_repression_multiplier=1.0,
        gradient_strength=0.0,
        overlap_penalty=False,
        seed=seed,
    )
    return gen.generate()


def generate_trap_standard(seed=0):
    gen = BenchmarkDataGenerator(
        n_cells=2000, n_genes=2000, n_cell_types=8, n_patterns=3,
        dropout=0.1, noise=0.15, effect=0.7,
        n_active_per_pattern=30, n_decoys=0,
        max_mirnas=50, min_targets=5,
        baseline_mode=False, signal_protection=0.5,
        machinery_level=150.0, background_level=30.0,
        target_background=100.0,
        target_repressed=2.0,
        true_mirnas_per_pattern=12,
        easy_negative_ratio=0.45,
        n_antifallback_cells=0,
        cis_repress_factor=0.15,
        key_mirna_fraction=0.35,
        key_repression_multiplier=0.1,
        support_repression_multiplier=4.0,
        gradient_strength=0.6,
        overlap_penalty=True,
        ortho_min_targets=20,
        ortho_max_targets=500,
        seed=seed
    )
    return gen.generate()


def generate_trap_small(seed=0):
    gen = BenchmarkDataGenerator(
        n_cells=500, n_genes=1500, n_cell_types=4, n_patterns=4,
        dropout=0.1, noise=0.15, effect=0.8,
        n_active_per_pattern=20, n_decoys=0,
        max_mirnas=50, min_targets=5,
        baseline_mode=False, signal_protection=0.5,
        machinery_level=150.0, background_level=30.0,
        target_background=100.0, target_repressed=2.0,
        true_mirnas_per_pattern=10,
        easy_negative_ratio=0.45,
        n_antifallback_cells=0,
        cis_repress_factor=0.15,
        key_mirna_fraction=0.35,
        key_repression_multiplier=0.1,
        support_repression_multiplier=4.0,
        gradient_strength=0.6,
        overlap_penalty=True,
        ortho_min_targets=20,
        ortho_max_targets=500,
        seed=seed
    )
    return gen.generate()


def generate_baseline_small(seed=0):
    gen = BenchmarkDataGenerator(
        n_cells=500, n_genes=1500, n_cell_types=4, n_patterns=4,
        dropout=0.15, noise=0.2, effect=0.8,
        n_active_per_pattern=30, n_decoys=0,
        max_mirnas=30, min_targets=5,
        baseline_mode=False,
        signal_protection=0.3, machinery_level=150.0,
        background_level=40.0, target_background=80.0, target_repressed=5.0,
        true_mirnas_per_pattern=7,
        easy_negative_ratio=0.90,
        n_antifallback_cells=0,
        cis_repress_factor=0.15,
        key_mirna_fraction=1.0,
        key_repression_multiplier=1.0,
        support_repression_multiplier=1.0,
        gradient_strength=0.0,
        overlap_penalty=False,
        seed=seed,
    )
    return gen.generate()


def generate_negative_control(seed=0):
    gen = BenchmarkDataGenerator(
        n_cells=1000, n_genes=1500, n_cell_types=4, n_patterns=0,
        dropout=0.2, noise=0.3, effect=0.0, n_active_per_pattern=0, n_decoys=0,
        baseline_mode=True, signal_protection=0.0, seed=seed
    )
    return gen.generate()


generate_clean_setting = generate_baseline_standard
generate_trap_setting = generate_trap_standard
generate_smoke_setting = generate_baseline_small
generate_smoke_trap_setting = generate_trap_small
generate_harder_setting = generate_trap_setting


__all__ = [
    'BenchmarkDataGenerator', 'RealMiRNADatabase',
    'generate_baseline_standard', 'generate_trap_standard', 'generate_trap_small',
    'generate_baseline_small', 'generate_clean_setting', 'generate_trap_setting',
    'generate_smoke_setting', 'generate_smoke_trap_setting', 'generate_negative_control',
    'generate_harder_setting',
]
