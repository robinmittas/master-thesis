# %% [markdown]
# # Integration of snRNA and scRNA data
#
# In this notebook snRNA and scRNA data is integrated into one latent space. To construct a common latent space for both measurements we are using `scVI` in this notebook
#
# **Requires:**
# - `/vol/storage/data/pancreas_multiome/processed/gex_e14.5.h5ad`
# - `/vol/storage/data/pancreas_sc/processed/gex_e14.5.h5ad`
# - `/vol/storage/data/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz` (downloadable from ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz)
#
# **Output:**
# - `/vol/storage/data/pancreas_sc_multiome/sn_sc_rna_scvi.h5ad`

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

import mplscience
import scvi

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv

sys.path.append("../..")
from paths import FIG_DIR, PROJECT_DIR  # isort: skip  # noqa: E402

# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = True
SAVE_MODEL = True
SN_PROCESSED_DIR = PROJECT_DIR / "pancreas_multiome" / "processed"
SC_PROCESSED_DIR = PROJECT_DIR / "pancreas_sc" / "processed"

sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

celltype_colors = {
    "celltype_colors": {
        "Alpha": "#1f78b4",
        "Beta": "#b2df8a",
        "Delta": "#6a3d9a",
        "Ductal": "#8fbc8f",
        "Epsilon": "#cab2d6",
        "Ngn3 high EP": "#fdbf6f",
        "Ngn3 low EP": "#f4a460",
        "Pre-endocrine": "#ff7f00",
    },
    "celltype_fine_colors": {
        "Alpha": "#1f78b4",
        "Beta": "#b2df8a",
        "Delta": "#6a3d9a",
        "Ductal": "#8fbc8f",
        "Eps/Delta progenitors": "#029e73",
        "Epsilon": "#cab2d6",
        "Ngn3 high EP": "#fdbf6f",
        "Ngn3 low EP": "#f4a460",
        "Fev+ Alpha": "#d55e00",
        "Fev+ Beta": "#cc78bc",
        "Fev+ Delta": "#ca9161",
    },
}

protocol_colors = {"scRNA-seq": "#0173b2", "multiome": "#de8f05"}

# %% [markdown]
# ## Read and preprocess GEX data

# %%
adata_sn = sc.read(SN_PROCESSED_DIR / "gex_e14.5.h5ad")
adata_sc = sc.read(SC_PROCESSED_DIR / "gex_e14.5.h5ad")

adata = adata_sn.concatenate(adata_sc)
adata.layers["counts"] = adata.layers["spliced"] + adata.layers["unspliced"]

scv.pp.filter_genes(adata, min_counts=20)
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    layer="counts",
    batch_key="protocol",
    subset=True,
)
adata

# %% [markdown]
# ## Train scvi model to get common latent space

# %%
scvi.model.SCVI.setup_anndata(adata, batch_key="batch", layer="counts")  # model requires raw counts

vae = scvi.model.SCVI(adata, n_layers=4, n_latent=30, gene_likelihood="nb")  # Negative binomial distribution

# %%
vae.train(early_stopping=True)

# %% [markdown]
# ## Get latent representation and plot UMAP

# %%
# first plot unintegrated data
scv.pp.pca(adata)
scv.pp.neighbors(adata)
scv.tl.umap(adata)

adata.uns["celltype_colors"] = celltype_colors["celltype_colors"]
adata.uns["protocol_colors"] = protocol_colors

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="protocol", ax=ax, title="")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "data_integration"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "unintegrated_umap_protocol.svg", format="svg", transparent=True, bbox_inches="tight")


# %%
adata.obsm["X_scVI"] = vae.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=25)

sc.tl.umap(adata)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="celltype", ax=ax, legend_loc="right margin", title="")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "data_integration"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "integrated_umap_celltype.svg", format="svg", transparent=True, bbox_inches="tight")


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="protocol", ax=ax, title="")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "data_integration"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "integrated_umap_protocol.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Estimate abundances

# %%
# adata object still has raw counts, no need to reload
adata = scv.pp.estimate_abundance(
    adata,
    layers=["unspliced", "spliced"],
    mode="connectivities",
    dataset_key="protocol",
    sc_rna_name="scRNA-seq",
    min_estimation_samples=1,
    smooth_obs=False,  # dont smooth because moments basically do the same afterwards
    clip_cyto=True,
    lambda_correction=True,
    filter_zero_genes=True,
)

# %%
scv.pp.normalize_per_cell(
    adata,
    layers=["unspliced_nucleus", "spliced_nucleus", "spliced_cytoplasm", "spliced_cell"],
)

scv.pp.moments(
    adata,
    use_rep="X_scVI",
    layers={"unspliced_nucleus": "Mu_nuc", "spliced_nucleus": "Ms_nuc", "spliced_cytoplasm": "Ms_cyt"},
)

scaler = MinMaxScaler()
adata.layers["Mu_nuc"] = scaler.fit_transform(adata.layers["Mu_nuc"])

scaler = MinMaxScaler()
adata.layers["Ms_nuc"] = scaler.fit_transform(adata.layers["Ms_nuc"])

scaler = MinMaxScaler()
adata.layers["Ms_cyt"] = scaler.fit_transform(adata.layers["Ms_cyt"])

# %%
adata_sn.obs_names

# %% [markdown]
# ## Store result with latent representation

# %%
adata.obs_names = adata.obs_names.str[:-2]
adata.write(PROJECT_DIR / "pancreas_sc_multiome" / "sn_sc_rna_scvi.h5ad")
