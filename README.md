# mirCCC

**Predicting miRNA-mediated cell–cell communication from single-cell transcriptomics via graph-diffused proxies and self-supervised graph transformers**

mirCCC infers extracellular vesicle (EV)-derived miRNA-mediated cell–cell communication from standard scRNA-seq data. It estimates per-cell miRNA activity through signed graph diffusion on a miRNA–target bipartite network, scores sender/receiver capacity using EV biogenesis, sorting, and RISC pathway genes, and constructs per-miRNA directed communication graphs processed by a Graph Transformer trained via Deep Graph Infomax — without requiring any labeled communication data.

## Installation

```bash
git clone https://github.com/yfchen801/mirCCC.git
cd mirCCC
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- PyTorch Geometric ≥ 2.3
- CUDA-compatible GPU recommended (CPU also supported)

## Quick Start: Tutorial

See [`tutorial_pipeline.ipynb`](tutorial_pipeline.ipynb) for a complete walkthrough of mirCCC on real scRNA-seq data (Pelka et al. 2021, CRC atlas).

The tutorial covers:
1. Data loading and preprocessing (GSE178341)
2. Patient selection and quality control
3. Running the mirCCC pipeline
4. All 7 visualization functions:
   - `plot_chord` — global communication network
   - `plot_chord_focal` — cell-type-centric communication
   - `plot_sankey` — communication flow diagram
   - `plot_dotplot` — miRNA × axis communication matrix
   - `plot_circos` — miRNA–target gene circuit
   - `plot_mirna_ranking` — per-axis miRNA ranking
   - `plot_target_heatmap` — target gene expression by cell type

### Data

- **scRNA-seq**: Download from [GEO GSE178341](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341) and place in `./data/`
- **miRNA references**: Included in `./mirna/`

### Input

An AnnData object (`.h5ad`) with:

- `adata.X` — raw count matrix (cells × genes)
- `adata.obs['cell_type']` — cell type annotation
- `adata.var_names` — gene symbols (HGNC format preferred)

### Output

```
results/
├── model/                          # Trained model checkpoint
├── results/
│   ├── mirna_edges.csv             # Per-miRNA edge-level communication scores
│   ├── cellpair_communication.csv  # Cell-pair aggregated scores
│   └── celltype_matrix.csv         # Cell-type level communication matrix
└── figures/                        # Visualization outputs
```

## Pipeline

mirCCC runs in five steps:

1. **miRNA proxy inference** — signed graph diffusion on a miRNA–target bipartite network (miRTarBase) infers per-cell miRNA activity from mRNA expression
2. **Sender/Receiver scoring** — EV biogenesis + sorting genes define sender capacity; RISC pathway genes + target repression define receiver capacity
3. **Per-miRNA graph construction** — a directed communication graph is built independently for each miRNA
4. **Graph Transformer training** — a two-layer TransformerConv encoder is trained via Deep Graph Infomax (self-supervised)
5. **Communication scoring** — per-miRNA match score × machinery gate × learned attention, aggregated to cell-type level

## Visualization

mirCCC provides 7 built-in visualization functions via `mirCCC_viz.py`:

| Function | Description |
|----------|-------------|
| `plot_chord()` | Global chord diagram of miRNA-mediated CCC |
| `plot_chord_focal()` | Chord diagram filtered by a specific cell type |
| `plot_dotplot()` | EV-miRNA communication dot plot (size = EV release, color = strength) |
| `plot_sankey()` | Sender → Receiver communication flow |
| `plot_circos()` | miRNA–target gene circos network |
| `plot_mirna_ranking()` | miRNA ranking bar chart for a specific communication axis |
| `plot_target_heatmap()` | Z-scored target gene expression heatmap across cell types |

All functions accept `edge_df` (pipeline output) and `adata` as primary inputs. Pass `save_path='file.pdf'` to save figures.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_S_SENDERS` | 150 | Top sender cells selected per miRNA |
| `TOP_R_RECEIVERS` | 150 | Top receiver cell candidates |
| `TOP_L_PER_SENDER` | 10 | Receivers connected per sender |
| `HIDDEN_DIM` | 128 | Graph Transformer hidden dimension |
| `NUM_HEADS` | 4 | Attention heads |
| `MIR_EMB_DIM` | 16 | miRNA embedding dimension |
| `NUM_EPOCHS` | 500 | Training epochs |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `GAMMA_COV` | 2.5 | Coverage down-weighting exponent |

## License

MIT License. See [LICENSE](LICENSE) for details.
