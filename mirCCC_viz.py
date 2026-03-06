"""
mirCCC Visualization Module
============================
Publication-quality visualizations for miRNA-mediated cell-cell communication.

Usage:
    from mirCCC_viz import plot_chord, plot_chord_focal, plot_dotplot
    from mirCCC_viz import plot_sankey, plot_circos, plot_mirna_ranking, plot_target_heatmap

All functions take edge_df (pipeline output) + adata as primary inputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path as MPath
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import rankdata
import os


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _get_ct_colors(cell_types):
    palette = sns.color_palette("husl", len(cell_types))
    return dict(zip(cell_types, [mcolors.to_hex(c) for c in palette]))


def _norm_mir(s):
    return s.lower().replace('hsa-', '').replace('_', '-').strip()


def _build_comm_matrix(edge_df, value='count'):
    comm = {}
    for (s, r), grp in edge_df.groupby(['sender_type', 'receiver_type']):
        if value == 'count':
            comm[(s, r)] = len(grp)
        elif value == 'score':
            comm[(s, r)] = grp['score'].sum()
    return comm


# ══════════════════════════════════════════════════════════════
# 1. CHORD DIAGRAM
# ══════════════════════════════════════════════════════════════

def _draw_chord_core(comm_matrix, cell_types, ct_colors, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    n = len(cell_types)
    ct_total = {}
    for ct in cell_types:
        ct_total[ct] = sum(v for (s, r), v in comm_matrix.items() if s == ct or r == ct)
    grand_total = sum(ct_total.values())
    if grand_total == 0:
        return

    gap_deg = 3
    available_deg = 360 - gap_deg * n
    min_span_deg = available_deg * 0.08
    raw_spans = {ct: (ct_total[ct] / grand_total) * available_deg for ct in cell_types}
    needs_boost = {ct: s < min_span_deg for ct, s in raw_spans.items()}
    boosted_total = sum(min_span_deg for ct in cell_types if needs_boost[ct])
    remaining_deg = available_deg - boosted_total
    large_total = sum(ct_total[ct] for ct in cell_types if not needs_boost[ct])

    arc_spans = {}
    start = 90
    for ct in cell_types:
        if needs_boost[ct]:
            span = min_span_deg
        else:
            span = (ct_total[ct] / large_total) * remaining_deg if large_total > 0 else min_span_deg
        arc_spans[ct] = (start, start + span)
        start += span + gap_deg

    R_outer, R_inner, R_label = 1.0, 0.90, 1.15
    Rr = R_inner - 0.08

    for ct in cell_types:
        a0, a1 = arc_spans[ct]
        theta = np.linspace(np.radians(a0), np.radians(a1), 100)
        xs_o = R_outer * np.cos(theta)
        ys_o = R_outer * np.sin(theta)
        xs_i = R_inner * np.cos(theta[::-1])
        ys_i = R_inner * np.sin(theta[::-1])
        verts = list(zip(xs_o, ys_o)) + list(zip(xs_i, ys_i))
        verts.append(verts[0])
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
        patch = mpatches.PathPatch(MPath(verts, codes), facecolor=ct_colors[ct],
                                   edgecolor='white', linewidth=1.5, alpha=0.95, zorder=3)
        ax.add_patch(patch)
        mid = np.radians((a0 + a1) / 2)
        lx, ly = R_label * np.cos(mid), R_label * np.sin(mid)
        rot = np.degrees(mid) - 90
        if -270 < rot < -90:
            rot += 180
        ax.text(lx, ly, ct, ha='center', va='center', fontsize=13,
                fontweight='bold', rotation=rot, rotation_mode='anchor')

    ct_cursor = {ct: arc_spans[ct][0] for ct in cell_types}

    def alloc(ct, frac):
        span = arc_spans[ct][1] - arc_spans[ct][0]
        sub = span * frac
        s = ct_cursor[ct]
        ct_cursor[ct] = s + sub
        return s, s + sub

    connections = sorted(comm_matrix.items(), key=lambda x: -x[1])
    for (s, r), v in connections:
        if v <= 0:
            continue
        frac_s = v / ct_total[s] if ct_total[s] > 0 else 0
        frac_r = v / ct_total[r] if ct_total[r] > 0 else 0
        s_a0_deg, s_a1_deg = alloc(s, frac_s)
        r_a0_deg, r_a1_deg = alloc(r, frac_r)
        s_a0, s_a1 = np.radians(s_a0_deg), np.radians(s_a1_deg)
        r_a0, r_a1 = np.radians(r_a0_deg), np.radians(r_a1_deg)
        n_arc = 30
        color = ct_colors[s]

        th_s = np.linspace(s_a0, s_a1, n_arc)
        pts_s = [(Rr * np.cos(t), Rr * np.sin(t)) for t in th_s]
        th_r = np.linspace(r_a1, r_a0, n_arc)
        pts_r = [(Rr * np.cos(t), Rr * np.sin(t)) for t in th_r]

        verts, codes = [], []
        verts.append(pts_s[0]); codes.append(MPath.MOVETO)
        for p in pts_s[1:]:
            verts.append(p); codes.append(MPath.LINETO)
        verts.append((0, 0)); verts.append(pts_r[0])
        codes.append(MPath.CURVE3); codes.append(MPath.CURVE3)
        for p in pts_r[1:]:
            verts.append(p); codes.append(MPath.LINETO)
        verts.append((0, 0)); verts.append(pts_s[0])
        codes.append(MPath.CURVE3); codes.append(MPath.CURVE3)
        path = MPath(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none', alpha=0.45, zorder=2)
        ax.add_patch(patch)

        r_mid_angle = (r_a0 + r_a1) / 2
        arrow_verts = [
            (Rr * np.cos(r_a0), Rr * np.sin(r_a0)),
            (R_inner * np.cos(r_mid_angle), R_inner * np.sin(r_mid_angle)),
            (Rr * np.cos(r_a1), Rr * np.sin(r_a1)),
            (Rr * np.cos(r_a0), Rr * np.sin(r_a0)),
        ]
        arrow_codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MPath(arrow_verts, arrow_codes),
                                        facecolor=color, edgecolor='none', alpha=0.75, zorder=4))

        th_bar = np.linspace(s_a0, s_a1, 20)
        bar_outer = [(R_inner * np.cos(t), R_inner * np.sin(t)) for t in th_bar]
        bar_inner = [(Rr * np.cos(t), Rr * np.sin(t)) for t in th_bar[::-1]]
        bar_verts = bar_outer + bar_inner
        bar_verts.append(bar_verts[0])
        bar_codes = [MPath.MOVETO] + [MPath.LINETO] * (len(bar_verts) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MPath(bar_verts, bar_codes),
                                        facecolor=color, edgecolor='none', alpha=0.55, zorder=2))

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)


def plot_chord(edge_df, adata, value='count', title='miRNA-mediated Cell-Cell Communication',
               save_path=None, figsize=(10, 10)):
    cell_types = sorted(adata.obs['cell_type'].unique())
    ct_colors = _get_ct_colors(cell_types)
    comm = _build_comm_matrix(edge_df, value=value)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_chord_core(comm, cell_types, ct_colors, title=title, ax=ax)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 2. CHORD DIAGRAM — FOCAL CELL TYPE
# ══════════════════════════════════════════════════════════════

def plot_chord_focal(edge_df, adata, focal_celltype, value='count',
                     save_path=None, figsize=(10, 10)):
    cell_types = sorted(adata.obs['cell_type'].unique())
    ct_colors = _get_ct_colors(cell_types)
    comm_full = _build_comm_matrix(edge_df, value=value)
    comm_focal = {(s, r): v for (s, r), v in comm_full.items()
                  if s == focal_celltype or r == focal_celltype}
    fig, ax = plt.subplots(figsize=figsize)
    _draw_chord_core(comm_focal, cell_types, ct_colors,
                     title=f'{focal_celltype}-centric miRNA Communication', ax=ax)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. COMMUNICATION DOT PLOT
# ══════════════════════════════════════════════════════════════

def plot_dotplot(edge_df, adata, proxy_matrix=None, top_n_mirna=20,
                 title='EV-derived miRNA Communication Dot Plot',
                 save_path=None):
    agg = edge_df.groupby(['sender_type', 'receiver_type', 'mirna']).agg(
        comm_strength=('score', 'sum'),
        n_edges=('score', 'count'),
    ).reset_index()

    if proxy_matrix is not None:
        PR = proxy_matrix
        ct_labels = adata.obs['cell_type'].values
        pr_norm2col = {_norm_mir(c): c for c in PR.columns}
        ev_release = []
        for _, row in agg.iterrows():
            mirna_raw = row['mirna']
            col = None
            if mirna_raw in PR.columns:
                col = mirna_raw
            elif _norm_mir(mirna_raw) in pr_norm2col:
                col = pr_norm2col[_norm_mir(mirna_raw)]
            if col is None:
                ev_release.append(0.0)
                continue
            mask = ct_labels == row['sender_type']
            ev_release.append(np.mean(np.abs(PR.loc[mask, col].values)) if mask.sum() > 0 else 0.0)
        agg['ev_release'] = ev_release
    else:
        agg['ev_release'] = agg['n_edges'].astype(float)

    top_mirnas = agg.groupby('mirna')['comm_strength'].sum().nlargest(top_n_mirna).index.tolist()
    agg = agg[agg['mirna'].isin(top_mirnas)].copy()
    agg['pair'] = agg['sender_type'] + ' | ' + agg['receiver_type']

    pair_order = agg.groupby('pair')['comm_strength'].sum().sort_values(ascending=True).index.tolist()
    mirna_order = agg.groupby('mirna')['comm_strength'].sum().sort_values(ascending=False).index.tolist()
    mirna_labels = [m.replace('hsa-', '') for m in mirna_order]

    S_MIN, S_MAX = 12, 550
    if agg['ev_release'].max() - agg['ev_release'].min() < 1e-10:
        agg['_size'] = (S_MIN + S_MAX) / 2
    else:
        agg['_ev_rank'] = rankdata(agg['ev_release'].values, method='average') / len(agg)
        agg['_size'] = S_MIN + agg['_ev_rank'] * (S_MAX - S_MIN)

    n_x, n_y = len(mirna_order), len(pair_order)
    cell_w, cell_h = 0.55, 0.50
    margin_l, margin_r, margin_t, margin_b = 1.8, 3.0, 0.8, 2.0
    fig_w = margin_l + n_x * cell_w + margin_r
    fig_h = margin_t + n_y * cell_h + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_left = margin_l / fig_w
    ax_bottom = margin_b / fig_h
    ax_width = (n_x * cell_w) / fig_w
    ax_height = (n_y * cell_h) / fig_h
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    sc = None
    for _, row in agg.iterrows():
        if row['mirna'] not in mirna_order or row['pair'] not in pair_order:
            continue
        xi = mirna_order.index(row['mirna'])
        yi = pair_order.index(row['pair'])
        sc = ax.scatter(xi, yi, s=row['_size'], c=row['comm_strength'],
                        cmap='viridis', vmin=0, vmax=agg['comm_strength'].quantile(0.95),
                        edgecolors='none', linewidth=0, zorder=3)

    ax.set_xticks(range(n_x))
    ax.set_xticklabels(mirna_labels, rotation=60, ha='right', fontsize=9)
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(pair_order, fontsize=9)
    ax.set_xlim(-0.8, n_x - 0.2)
    ax.set_ylim(-0.8, n_y - 0.2)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_xlabel('EV-derived miRNAs', fontsize=12)
    ax.set_ylabel('Senders | Receivers', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    quantiles = [0.10, 0.35, 0.65, 0.90]
    q_sizes = [S_MIN + q * (S_MAX - S_MIN) for q in quantiles]
    size_handles = [ax.scatter([], [], s=qs, c='#555555', edgecolors='none', alpha=0.6,
                               label=f'{int(q*100)}%') for q, qs in zip(quantiles, q_sizes)]
    ax.legend(handles=size_handles, loc='upper left', bbox_to_anchor=(1.03, 1.0),
              title='EV release', fontsize=9, title_fontsize=10, framealpha=0.9,
              labelspacing=2.5, borderpad=1.2, handletextpad=1.5)

    cax_left = (margin_l + n_x * cell_w + 0.3) / fig_w
    cax_bottom = ax_bottom
    cax_width = 0.25 / fig_w
    cax_height = ax_height * 0.45
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('Comm. strength', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. SANKEY DIAGRAM
# ══════════════════════════════════════════════════════════════

def plot_sankey(edge_df, adata, title='miRNA-mediated Cell Communication Flow\n(Sender → Receiver)',
                save_path=None, figsize=(10, 8)):
    flow = edge_df.groupby(['sender_type', 'receiver_type']).agg(
        total_score=('score', 'sum')
    ).reset_index()

    cell_types_sorted = sorted(adata.obs['cell_type'].unique())
    ct_colors = _get_ct_colors(cell_types_sorted)

    send_totals = flow.groupby('sender_type')['total_score'].sum()
    recv_totals = flow.groupby('receiver_type')['total_score'].sum()
    send_order = send_totals.sort_values(ascending=False).index.tolist()
    recv_order = recv_totals.sort_values(ascending=False).index.tolist()
    grand = max(send_totals.sum(), recv_totals.sum())

    fig, ax = plt.subplots(figsize=figsize)
    x_left, x_right = 0.0, 1.0
    bar_w = 0.06
    gap_frac = 0.03
    usable_l = 1.0 - gap_frac * max(len(send_order) - 1, 0)
    usable_r = 1.0 - gap_frac * max(len(recv_order) - 1, 0)

    y_left = {}
    cursor = 0.0
    for ct in send_order:
        h = (send_totals[ct] / grand) * usable_l
        y_left[ct] = (cursor, cursor + h)
        cursor += h + gap_frac

    y_right = {}
    cursor = 0.0
    for ct in recv_order:
        h = (recv_totals[ct] / grand) * usable_r
        y_right[ct] = (cursor, cursor + h)
        cursor += h + gap_frac

    flow_sorted = flow.sort_values('total_score', ascending=False)
    left_cursor = {ct: y_left[ct][0] for ct in send_order}
    right_cursor = {ct: y_right[ct][0] for ct in recv_order}

    for _, row in flow_sorted.iterrows():
        s, r, val = row['sender_type'], row['receiver_type'], row['total_score']
        h_l = (val / grand) * usable_l
        h_r = (val / grand) * usable_r
        y0_l = left_cursor[s]
        y0_r = right_cursor[r]
        left_cursor[s] += h_l
        right_cursor[r] += h_r
        color = ct_colors.get(s, '#aaaaaa')

        t = np.linspace(0, 1, 80)
        x_b = (1 - t) ** 3 * (x_left + bar_w) + 3 * (1 - t) ** 2 * t * 0.35 + \
              3 * (1 - t) * t ** 2 * 0.65 + t ** 3 * (x_right - bar_w)
        y_top = (1 - t) ** 3 * (y0_l + h_l) + 3 * (1 - t) ** 2 * t * (y0_l + h_l) + \
                3 * (1 - t) * t ** 2 * (y0_r + h_r) + t ** 3 * (y0_r + h_r)
        y_bot = (1 - t) ** 3 * y0_l + 3 * (1 - t) ** 2 * t * y0_l + \
                3 * (1 - t) * t ** 2 * y0_r + t ** 3 * y0_r

        verts = list(zip(x_b, y_top)) + list(zip(x_b[::-1], y_bot[::-1]))
        verts.append(verts[0])
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                                        edgecolor='none', alpha=0.45, zorder=2))

    for ct in send_order:
        y0, y1 = y_left[ct]
        ax.add_patch(plt.Rectangle((x_left, y0), bar_w, y1 - y0,
                                   facecolor=ct_colors.get(ct, '#aaa'),
                                   edgecolor='white', linewidth=1.2, zorder=4))
        ax.text(x_left + bar_w / 2, (y0 + y1) / 2, ct, ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none'))

    for ct in recv_order:
        y0, y1 = y_right[ct]
        ax.add_patch(plt.Rectangle((x_right - bar_w, y0), bar_w, y1 - y0,
                                   facecolor=ct_colors.get(ct, '#aaa'),
                                   edgecolor='white', linewidth=1.2, zorder=4))
        ax.text(x_right - bar_w / 2, (y0 + y1) / 2, ct, ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, ec='none'))

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, max(cursor, 1.0) + 0.05)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 5. CIRCOS — miRNA ↔ TARGET GENE
# ══════════════════════════════════════════════════════════════

def plot_circos(edge_df, adata, mir2tar_df, sender_types=None, receiver_types=None,
                top_n_mirna=12, top_n_target_per_mirna=5,
                title=None, save_path=None, figsize=(14, 14)):

    mask = pd.Series(True, index=edge_df.index)
    if sender_types is not None:
        mask &= edge_df['sender_type'].isin(sender_types)
    if receiver_types is not None:
        mask &= edge_df['receiver_type'].isin(receiver_types)
    focal_edges = edge_df[mask]

    if len(focal_edges) == 0:
        print("No edges match the specified sender/receiver types.")
        return

    col_mi = next((c for c in ['miRNA', 'mirna', 'miR'] if c in mir2tar_df.columns), None)
    col_tg = next((c for c in ['target_gene', 'target', 'gene'] if c in mir2tar_df.columns), None)

    mir_agg = focal_edges.groupby(['mirna', 'sender_type'])['score'].sum().reset_index()
    mir_best = mir_agg.sort_values('score', ascending=False).drop_duplicates('mirna')
    mir_best = mir_best.nlargest(top_n_mirna, 'score')
    top_mir = mir_best['mirna'].tolist()
    mir_sender_map = dict(zip(mir_best['mirna'], mir_best['sender_type']))
    mir_scores = dict(zip(mir_best['mirna'], mir_best['score']))

    mir_receiver_map = {}
    for m in top_mir:
        sub = focal_edges[focal_edges['mirna'] == m]
        mir_receiver_map[m] = sub.groupby('receiver_type')['score'].sum().idxmax()

    var_upper = set(adata.var_names.str.upper())
    mir_target_links = []
    for mirna in top_mir:
        pat = _norm_mir(mirna)
        m_mask = mir2tar_df[col_mi].apply(lambda x: _norm_mir(str(x)) == pat)
        tgts = mir2tar_df.loc[m_mask, col_tg].unique()
        tgts_in = [g for g in tgts if str(g).upper() in var_upper][:top_n_target_per_mirna]
        score = mir_scores[mirna]
        sender_ct = mir_sender_map[mirna]
        recv_ct = mir_receiver_map[mirna]
        for g in tgts_in:
            mir_target_links.append({
                'mirna': mirna, 'mir_label': f"{mirna} ({sender_ct})",
                'target': g, 'tgt_label': f"{g} ({recv_ct})",
                'score': score / max(len(tgts_in), 1),
                'sender_type': sender_ct, 'receiver_type': recv_ct,
            })

    if len(mir_target_links) == 0:
        print("No miRNA-target links found.")
        return

    link_df = pd.DataFrame(mir_target_links)
    mirnas_used = link_df.groupby('mir_label')['score'].sum().sort_values(ascending=False).index.tolist()
    targets_used = link_df.groupby('tgt_label')['score'].sum().sort_values(ascending=False).index.tolist()
    mir_totals = link_df.groupby('mir_label')['score'].sum()
    tgt_totals = link_df.groupby('tgt_label')['score'].sum()
    n_mir, n_tgt = len(mirnas_used), len(targets_used)

    GAP, GAP_MINOR, MIN_ARC = 8, 1.0, 2.0
    mir_zone_start, mir_zone_end = -90 + GAP, 90 - GAP
    tgt_zone_start, tgt_zone_end = 90 + GAP, 270 - GAP

    mir_avail = (mir_zone_end - mir_zone_start) - GAP_MINOR * max(n_mir - 1, 0)
    mir_grand = mir_totals.sum()
    mir_arcs = {}
    cursor = mir_zone_start
    for m in mirnas_used:
        span = max((mir_totals[m] / mir_grand) * mir_avail, MIN_ARC)
        mir_arcs[m] = (cursor, cursor + span)
        cursor += span + GAP_MINOR

    tgt_avail = (tgt_zone_end - tgt_zone_start) - GAP_MINOR * max(n_tgt - 1, 0)
    tgt_grand = tgt_totals.sum()
    tgt_arcs = {}
    cursor = tgt_zone_start
    for g in targets_used:
        span = max((tgt_totals[g] / tgt_grand) * tgt_avail, MIN_ARC)
        tgt_arcs[g] = (cursor, cursor + span)
        cursor += span + GAP_MINOR

    all_senders = sorted(link_df['sender_type'].unique())
    all_receivers = sorted(link_df['receiver_type'].unique())
    send_palette = dict(zip(all_senders, [mcolors.to_hex(c) for c in sns.color_palette("Set2", len(all_senders))]))
    recv_palette = dict(zip(all_receivers, [mcolors.to_hex(c) for c in sns.color_palette("Set1", len(all_receivers))]))
    label_to_sender = dict(zip(link_df['mir_label'], link_df['sender_type']))
    label_to_receiver = dict(zip(link_df['tgt_label'], link_df['receiver_type']))
    mir_colors = {ml: send_palette[label_to_sender[ml]] for ml in mirnas_used}
    tgt_colors = {tl: recv_palette[label_to_receiver[tl]] for tl in targets_used}

    R_outer, R_inner, R_link, R_label = 1.0, 0.88, 0.86, 1.04

    fig, ax = plt.subplots(figsize=figsize)

    def draw_arc(a0, a1, r_out, r_in, color, alpha=0.9):
        n_pts = max(int(abs(a1 - a0) * 2), 10)
        theta = np.linspace(np.radians(a0), np.radians(a1), n_pts)
        xo, yo = r_out * np.cos(theta), r_out * np.sin(theta)
        xi, yi = r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])
        verts = list(zip(xo, yo)) + list(zip(xi, yi))
        verts.append(verts[0])
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                                        edgecolor='white', linewidth=0.5, alpha=alpha, zorder=3))

    def radial_label(angle_deg, radius, text, fontsize=7, color='black', fontweight='normal'):
        a = np.radians(angle_deg)
        x, y = radius * np.cos(a), radius * np.sin(a)
        rot = angle_deg
        if 90 < angle_deg % 360 < 270:
            rot += 180; ha = 'right'
        else:
            ha = 'left'
        ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, fontweight=fontweight,
                color=color, rotation=rot, rotation_mode='anchor')

    for m in mirnas_used:
        a0, a1 = mir_arcs[m]
        draw_arc(a0, a1, R_outer, R_inner, mir_colors[m])
        radial_label((a0 + a1) / 2, R_label, m, fontsize=8, fontweight='bold', color=mir_colors[m])

    for g in targets_used:
        a0, a1 = tgt_arcs[g]
        draw_arc(a0, a1, R_outer, R_inner, tgt_colors[g], alpha=0.75)
        radial_label((a0 + a1) / 2, R_label, g, fontsize=6.5, color=tgt_colors[g])

    mir_cursor = {m: mir_arcs[m][0] for m in mirnas_used}
    tgt_cursor = {g: tgt_arcs[g][0] for g in targets_used}
    max_sc = link_df['score'].max()

    for _, row in link_df.sort_values('score', ascending=False).iterrows():
        ml, tl, sc_val = row['mir_label'], row['tgt_label'], row['score']
        m_span = mir_arcs[ml][1] - mir_arcs[ml][0]
        m_sub = m_span * (sc_val / mir_totals[ml]) if mir_totals[ml] > 0 else 0
        m_a0 = mir_cursor[ml]; m_a1 = m_a0 + m_sub; mir_cursor[ml] = m_a1
        g_span = tgt_arcs[tl][1] - tgt_arcs[tl][0]
        g_sub = g_span * (sc_val / tgt_totals[tl]) if tgt_totals[tl] > 0 else 0
        g_a0 = tgt_cursor[tl]; g_a1 = g_a0 + g_sub; tgt_cursor[tl] = g_a1

        color = mir_colors[ml]
        alpha = np.clip(sc_val / max_sc * 0.5 + 0.15, 0.1, 0.6)
        n_arc = 20
        th_m = np.linspace(np.radians(m_a0), np.radians(m_a1), n_arc)
        pts_m = [(R_link * np.cos(t), R_link * np.sin(t)) for t in th_m]
        th_g = np.linspace(np.radians(g_a1), np.radians(g_a0), n_arc)
        pts_g = [(R_link * np.cos(t), R_link * np.sin(t)) for t in th_g]

        verts, codes = [], []
        verts.append(pts_m[0]); codes.append(MPath.MOVETO)
        for p in pts_m[1:]: verts.append(p); codes.append(MPath.LINETO)
        verts.append((0, 0)); verts.append(pts_g[0]); codes.extend([MPath.CURVE3, MPath.CURVE3])
        for p in pts_g[1:]: verts.append(p); codes.append(MPath.LINETO)
        verts.append((0, 0)); verts.append(pts_m[0]); codes.extend([MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                                        edgecolor='none', alpha=alpha, zorder=2))

    legend_handles = []
    for ct, c in send_palette.items():
        legend_handles.append(mpatches.Patch(color=c, alpha=0.8, label=f'{ct} (sender)'))
    for ct, c in recv_palette.items():
        legend_handles.append(mpatches.Patch(color=c, alpha=0.7, label=f'{ct} (receiver)'))
    ax.legend(handles=legend_handles, loc='lower center', fontsize=10, framealpha=0.9,
              ncol=min(len(legend_handles), 4), bbox_to_anchor=(0.5, -0.03))

    lim = 1.65
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.axis('off')

    sender_str = ', '.join(sender_types) if sender_types else 'All'
    receiver_str = ', '.join(receiver_types) if receiver_types else 'All'
    if title is None:
        title = (f'miRNA–Target Gene Communication Network\n'
                 f'Sender: {sender_str} → Receiver: {receiver_str}')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 6. miRNA RANKING BAR CHART
# ══════════════════════════════════════════════════════════════

def plot_mirna_ranking(edge_df, sender_type, receiver_type, focal_mirnas=None,
                       focal_colors=None, top_n=25,
                       title=None, save_path=None, figsize=(8, 7)):
    axis_edges = edge_df[
        (edge_df['sender_type'] == sender_type) &
        (edge_df['receiver_type'] == receiver_type)
    ]
    if len(axis_edges) == 0:
        print(f"No edges found for {sender_type} → {receiver_type}")
        return

    mirna_rank = axis_edges.groupby('mirna').agg(
        total_score=('score', 'sum'),
        n_edges=('score', 'count'),
    ).reset_index().sort_values('total_score', ascending=False)
    mirna_rank['rank'] = range(1, len(mirna_rank) + 1)

    if focal_mirnas is None:
        focal_mirnas = []
    if focal_colors is None:
        focal_colors = {}

    def _flag(name):
        for pat in focal_mirnas:
            if _norm_mir(name) == _norm_mir(pat):
                return pat
        return None

    mirna_rank['focal'] = mirna_rank['mirna'].apply(_flag)

    top = mirna_rank.head(min(top_n, len(mirna_rank))).iloc[::-1]
    n_show = len(top)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [focal_colors.get(f, '#AED6F1') if f else '#AED6F1' for f in top['focal']]
    ax.barh(range(n_show), top['total_score'], color=colors,
            edgecolor='white', linewidth=0.5, height=0.72)
    ylabels = [f"{_norm_mir(r['mirna'])}  (n={r['n_edges']:.0f})" for _, r in top.iterrows()]
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel('Total communication score', fontsize=10)

    if title is None:
        title = f'{sender_type} → {receiver_type}: miRNA ranking (mirCCC)'
    ax.set_title(title, fontsize=12, fontweight='bold')

    for i, (_, row) in enumerate(top.iterrows()):
        if row['focal']:
            label = f"★ {row['focal']}  (rank #{row['rank']}/{len(mirna_rank)})"
            ax.text(row['total_score'] + 0.05, i, label, va='center', fontsize=8,
                    fontweight='bold', color=focal_colors.get(row['focal'], '#333'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 7. TARGET GENE HEATMAP (Z-SCORED)
# ══════════════════════════════════════════════════════════════

def plot_target_heatmap(adata, target_genes, gene_mirna_map=None, key_genes=None,
                        highlight_celltype=None,
                        title='Target gene expression by cell type',
                        save_path=None):
    present = [g for g in target_genes if g in adata.var_names]
    if len(present) < 3:
        print("Too few target genes found in adata — skipping.")
        return

    ct_list = sorted(adata.obs['cell_type'].unique())
    X_tgt = adata[:, present].X
    if sp.issparse(X_tgt):
        X_tgt = X_tgt.toarray()

    expr_mat = pd.DataFrame(index=present, columns=ct_list, dtype=float)
    for ct in ct_list:
        mask = adata.obs['cell_type'].values == ct
        expr_mat[ct] = X_tgt[mask].mean(axis=0)

    expr_z = expr_mat.subtract(expr_mat.mean(axis=1), axis=0)
    expr_z = expr_z.divide(expr_mat.std(axis=1).replace(0, 1), axis=0)

    if gene_mirna_map is None:
        gene_mirna_map = {}
    if key_genes is None:
        key_genes = set()
    else:
        key_genes = {g.upper() for g in key_genes}

    row_labels = []
    row_colors = []
    for g in present:
        mirs = gene_mirna_map.get(g, set())
        mir_str = '+'.join(sorted(mirs)) if mirs else ''
        if g.upper() in key_genes:
            row_labels.append(f"★ {g}  [{mir_str}]")
            row_colors.append('#E74C3C')
        else:
            row_labels.append(f"{g}  [{mir_str}]" if mir_str else g)
            row_colors.append('#999999')

    col_colors = []
    for ct in ct_list:
        if highlight_celltype and ct == highlight_celltype:
            col_colors.append('#E74C3C')
        else:
            col_colors.append('#555555')

    fig, ax = plt.subplots(figsize=(10, max(6, len(present) * 0.32)))
    sns.heatmap(expr_z, cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.3, linecolor='white',
                yticklabels=row_labels,
                cbar_kws={'label': 'Z-score (expression)', 'shrink': 0.6})

    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(row_colors[i])
        tick.set_fontsize(7)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(col_colors[i])
        if highlight_celltype and ct_list[i] == highlight_celltype:
            tick.set_fontweight('bold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Target Gene')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
