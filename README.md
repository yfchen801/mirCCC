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

## Quick Start

### Run pipeline

```python
import scanpy as sc
from mirCCC_integrated_pipeline_api import Config, run_full_pipeline

adata = sc.read_h5ad("your_data.h5ad")

config = Config()
config.OUTPUT_DIR = "./results"
config.DEVICE = "cuda:0"  # or "cpu"
results = run_full_pipeline(config)
```

### Visualize results

```python
from mirCCC_viz import plot_chord, plot_sankey, plot_dotplot, plot_circos

edge_df = results['edge_df']

# Global communication network
plot_chord(edge_df, adata, save_path='chord.pdf')

# Sender → Receiver flow
plot_sankey(edge_df, adata, save_path='sankey.pdf')

# EV-miRNA communication dot plot
plot_dotplot(edge_df, adata, proxy_matrix=results['proxy_matrix'], save_path='dotplot.pdf')

# miRNA–target gene circos plot
import pandas as pd
mir2tar = pd.read_csv("data/mir2tar.csv")
plot_circos(edge_df, adata, mir2tar,
            sender_types=['Tumor'], receiver_types=['T/NK'],
            save_path='circos.pdf')
```

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

## Citation

```bibtex
@article{chen2026mirCCC,
  title={mirCCC: predicting miRNA-mediated cell--cell communication from
         single-cell transcriptomics via graph-diffused proxies and
         self-supervised graph transformers},
  author={Chen, Yifan},
  journal={Briefings in Bioinformatics},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
