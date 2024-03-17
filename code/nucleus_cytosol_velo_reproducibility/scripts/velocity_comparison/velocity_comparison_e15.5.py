# %% [markdown]
# # Velocity comparison - Pancreas E15.5
# In this notebook we want to compare if and how the Nuc/Cyt model compared with the original veloVI model solely trained on either snRNA or scRNA data infers RNA velocity and how the estimated velocities correlate. We further train one model on scRNA data which estimates the abundance within the whole cell of the snRNA-seq data.
# In this notebook we use the integrated data of the trained `scglue` (GEX) and `scVI` models to estimate abundances and for calculations where we require a neighbor graph.
# We further compare different velocity modes within this notebook for the nuc-cyt model
#
#
# **Requires:**
# - `sn_sc_rna_scvi.h5ad`
# - `sn_sc_rna_scglue.h5ad`
#
#    (Notebooks: `/notebooks/integration_imputation/`)
# - `pancreas_multiome/processed/gex_e14.5.h5ad`
#
#    (Notebook: `/notebooks/data_preprocessing/sn_rna_preprocess_pancreas.ipynb`)
# - `pancreas_sc/processed/gex_e14.5.h5ad`
#
#    (Notebook: `/notebooks/data_preprocessing/sc_rna_preprocess_pancreas.ipynb`)
#
# **Output:**
# This notebook will output different plots if `SAVE_FIGURES=TRUE`

# %% [markdown]
# ## Library imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

import mplscience
import torch
from velovi import VELOVI

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
from scvelo.inference import fit_velovi

sys.path.append("../..")
from paths import FIG_DIR, PROJECT_DIR  # isort: skip  # noqa: E402


# %% [markdown]
# ## Function Defintions
#


# %%
def compute_confidence(adata, vkey="velocities_velovi"):
    """Computes the velocity confidence of model.

    Parameters
    ----------
    adata
        Annotated data
    vkey
        adata.layers[vkey] should contain inferred velocities
    """
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=8)
    scv.tl.velocity_confidence(adata, vkey=vkey)

    g_df = pd.DataFrame()
    g_df["Velocity confidence"] = adata.obs[f"{vkey}_confidence"].to_numpy().ravel()

    return g_df


# %%
def _add_significance(ax, left: int, right: int, significance: str, level: int = 0, **kwargs):
    bracket_level = kwargs.pop("bracket_level", 1)
    bracket_height = kwargs.pop("bracket_height", 0.02)
    text_height = kwargs.pop("text_height", 0.01)

    bottom, top = ax.get_ylim()
    y_axis_range = top - bottom

    bracket_level = (y_axis_range * 0.07 * level) + top * bracket_level
    bracket_height = bracket_level - (y_axis_range * bracket_height)

    ax.plot([left, left, right, right], [bracket_height, bracket_level, bracket_level, bracket_height], **kwargs)

    ax.text(
        (left + right) * 0.5,
        bracket_level + (y_axis_range * text_height),
        significance,
        ha="center",
        va="bottom",
        c="k",
    )


# %%
def _get_significance(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# %%
def fit_velovi_(bdata):
    """Training function for original veloVI model for scRNA or snRNA data.

    Parameters
    ----------
    bdata
        Annotated data
    """
    VELOVI.setup_anndata(bdata, spliced_layer="Ms", unspliced_layer="Mu")

    vae = VELOVI(bdata)
    vae.train(max_epochs=500)

    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)

    bdata.layers["velocities_velovi"] = velocities / scaling
    bdata.layers["latent_time_velovi"] = latent_time

    bdata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    bdata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    bdata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    bdata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
    ) * scaling
    bdata.layers["fit_t"] = latent_time.values * np.expand_dims(scaling, axis=0)
    bdata.var["fit_scaling"] = scaling

    return vae, bdata


# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = True

sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

celltype_colors = {
    "Alpha": "#1f78b4",
    "Beta": "#b2df8a",
    "Delta": "#6a3d9a",
    "Ductal": "#8fbc8f",
    "Epsilon": "#cab2d6",
    "Ngn3 high EP": "#fdbf6f",
    "Ngn3 low EP": "#f4a460",
    "Pre-endocrine": "#ff7f00",
}

# %% [markdown]
# ## Read and preprocess multi-modal Data
#

# %%
# We will here compare scVI and scglue based estimated datasets
adata = sc.read(PROJECT_DIR / "pancreas_sc_multiome" / "sn_sc_rna_scglue_e15.5.h5ad")
adata_scvi = sc.read(PROJECT_DIR / "pancreas_sc_multiome" / "sn_sc_rna_scvi_e15.5.h5ad")
adata

# %% [markdown]
# ## Preprocess snRNA, scRNA data for later comparison

# %%
adata_sc = sc.read(PROJECT_DIR / "pancreas_sc" / "processed" / "gex_e15.5.h5ad")
adata_sn = sc.read(PROJECT_DIR / "pancreas_multiome" / "processed" / "gex_e15.5.h5ad")

# extract highly variable genes and normalize count data
scv.pp.filter_and_normalize(adata_sc, min_counts=20, n_top_genes=2000)
scv.pp.filter_and_normalize(adata_sn, min_counts=20, n_top_genes=2000)

# calculate neighbor graphs
scv.pp.neighbors(adata_sn)
scv.pp.neighbors(adata_sc)

# snRNA
scv.pp.moments(
    adata_sn,
    use_rep="X_pca",
)

scaler = MinMaxScaler()
adata_sn.layers["Mu"] = scaler.fit_transform(adata_sn.layers["Mu"])

scaler = MinMaxScaler()
adata_sn.layers["Ms"] = scaler.fit_transform(adata_sn.layers["Ms"])

## scRNA
scv.pp.moments(
    adata_sc,
    use_rep="X_pca",
)

scaler = MinMaxScaler()
adata_sc.layers["Mu"] = scaler.fit_transform(adata_sc.layers["Mu"])

scaler = MinMaxScaler()
adata_sc.layers["Ms"] = scaler.fit_transform(adata_sc.layers["Ms"])

# %% [markdown]
# ## Fit veloVI Nucleus/ Cytosol Model
# Note that we decrease the lr compared to original veloVI, as we might run into NaN issues when inferring mean, var for latent representation

# %%
vae_sn_sc, adata = fit_velovi(
    adata, max_epochs=500, unspliced_layer_nuc="Mu_nuc", spliced_layer_nuc="Ms_nuc", spliced_layer_cyt="Ms_cyt", lr=5e-3
)

# %%
vae_scvi, adata_scvi = fit_velovi(
    adata_scvi,
    max_epochs=500,
    unspliced_layer_nuc="Mu_nuc",
    spliced_layer_nuc="Ms_nuc",
    spliced_layer_cyt="Ms_cyt",
    lr=5e-3,
)

# %% [markdown]
# #### Plot losses

# %%
df = vae_sn_sc.history["elbo_train"].iloc[20:].reset_index().rename(columns={"elbo_train": "elbo"})
df["set"] = "train"

_df = vae_sn_sc.history["elbo_validation"].iloc[20:].reset_index().rename(columns={"elbo_validation": "elbo"})
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="elbo", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

df = (
    vae_sn_sc.history["reconstruction_loss_train"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_train": "reconstruction_loss"})
)
df["set"] = "train"

_df = (
    vae_sn_sc.history["reconstruction_loss_validation"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_validation": "reconstruction_loss"})
)
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="reconstruction_loss", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

# %% [markdown]
# ## Fit original veloVI model on scRNA-seq data!

# %%
vae_sc, adata_sc = fit_velovi_(adata_sc)

# %% [markdown]
# #### Plot losses

# %%
df = vae_sc.history["elbo_train"].iloc[20:].reset_index().rename(columns={"elbo_train": "elbo"})
df["set"] = "train"

_df = vae_sc.history["elbo_validation"].iloc[20:].reset_index().rename(columns={"elbo_validation": "elbo"})
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="elbo", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

df = (
    vae_sc.history["reconstruction_loss_train"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_train": "reconstruction_loss"})
)
df["set"] = "train"

_df = (
    vae_sc.history["reconstruction_loss_validation"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_validation": "reconstruction_loss"})
)
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="reconstruction_loss", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

# %% [markdown]
# ## Fit original veloVI model just on snRNA-seq data

# %%
vae_sn, adata_sn = fit_velovi_(adata_sn)

# %% [markdown]
# #### Plot losses

# %%
df = vae_sn.history["elbo_train"].iloc[20:].reset_index().rename(columns={"elbo_train": "elbo"})
df["set"] = "train"

_df = vae_sn.history["elbo_validation"].iloc[20:].reset_index().rename(columns={"elbo_validation": "elbo"})
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="elbo", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

df = (
    vae_sn.history["reconstruction_loss_train"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_train": "reconstruction_loss"})
)
df["set"] = "train"

_df = (
    vae_sn.history["reconstruction_loss_validation"]
    .iloc[20:]
    .reset_index()
    .rename(columns={"reconstruction_loss_validation": "reconstruction_loss"})
)
_df["set"] = "validation"

df = pd.concat([df, _df], axis=0).reset_index(drop=True)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="epoch", y="reconstruction_loss", hue="set", palette=["#0173B2", "#DE8F05"], ax=ax)
    plt.show()

# %% [markdown]
# ## Sample velocities with different `velo_mode`

# %%
# Define colors for celltype and cell cycle phases
adata.uns["celltype_colors"] = celltype_colors
adata_sc.uns["celltype_colors"] = celltype_colors
adata_sn.uns["celltype_colors"] = celltype_colors
adata_scvi.uns["celltype_colors"] = celltype_colors

# %% [markdown]
# ## Calculate velocities for all velo modes

# %%
velocities = vae_sn_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_cyt")
adata.layers["velocities_velovi_s_cyt"] = velocities
velocities = vae_sn_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_nuc")
adata.layers["velocities_velovi_s_nuc"] = velocities
velocities = vae_sn_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced_nuc")
adata.layers["velocities_velovi_u_nuc"] = velocities

velocities = vae_scvi.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_cyt")
adata_scvi.layers["velocities_velovi_s_cyt"] = velocities
velocities = vae_scvi.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_nuc")
adata_scvi.layers["velocities_velovi_s_nuc"] = velocities
velocities = vae_scvi.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced_nuc")
adata_scvi.layers["velocities_velovi_u_nuc"] = velocities

velocities = vae_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced")
adata_sc.layers["velocities_velovi"] = velocities
velocities = vae_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")
adata_sc.layers["velocities_velovi_u"] = velocities


velocities = vae_sn.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced")
adata_sn.layers["velocities_velovi"] = velocities
velocities = vae_sn.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")
adata_sn.layers["velocities_velovi_u"] = velocities

# %% [markdown]
# ## Velo confidence
# `scv.tl.velocity_confidence` requires layers "Ms" to be present --> we will set it to Ms_cyt or Ms_nuc respectivley (as we might use `velocities_velovi_s_nuc` as our velocity vector instead of `velocities_velovi_s_cyt`, see later phase portraits)

# %%
dfs = []

adata.layers["Ms"] = adata.layers["Ms_cyt"]
g_df = compute_confidence(adata, "velocities_velovi_s_cyt")
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model (s_cyt velo)"
dfs.append(g_df)

adata.layers["Ms"] = adata.layers["Ms_nuc"]
g_df = compute_confidence(adata, "velocities_velovi_s_nuc")
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model (s_nuc velo)"
dfs.append(g_df)

g_df = compute_confidence(adata_sc)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sc-model"
dfs.append(g_df)

g_df = compute_confidence(adata_sn)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sn-model"
dfs.append(g_df)

conf_df = pd.concat(dfs, axis=0)

# %%
g_df = compute_confidence(adata_sc)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sc-model"
dfs.append(g_df)

g_df = compute_confidence(adata_sn)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sn-model"
dfs.append(g_df)

conf_df = pd.concat(dfs, axis=0)

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 6))
    sns.violinplot(
        data=conf_df,
        ax=ax,
        orient="h",
        y="Dataset",
        x="Velocity confidence",
        hue="Method",
        palette=sns.color_palette("colorblind").as_hex()[:5],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([0.25, 0.5, 0.75, 1.0])
    plt.show()

# %% [markdown]
# ## Plot Velo Embedding for all 4 models

# %% [markdown]
# #### Nuc-cyt umap scvi

# %%
adata_scvi.uns["celltype_colors"] = celltype_colors
scv.tl.velocity_graph(adata_scvi, vkey="velocities_velovi_s_nuc", n_jobs=8, xkey="Ms_nuc")
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_scvi,
        vkey="velocities_velovi_s_nuc",
        color=["celltype"],
        cmap="viridis",
        legend_loc=False,
        title="",
        ax=ax,
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_snuc_scvi_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_snuc_scvi_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )

# %%
scv.tl.velocity_graph(adata_scvi, vkey="velocities_velovi_s_cyt", n_jobs=8, xkey="Ms_cyt")
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_scvi,
        vkey="velocities_velovi_s_cyt",
        color=["celltype"],
        cmap="viridis",
        legend_loc=False,
        title="",
        ax=ax,
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_scyt_scvi_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_scyt_scvi_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )

# %%
velocities = vae_scvi.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_sum")
adata_scvi.layers["velocities_velovi_s_sum"] = velocities
scv.tl.velocity_graph(adata_scvi, vkey="velocities_velovi_s_sum", n_jobs=8, xkey="Ms_sum")

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_scvi,
        vkey="velocities_velovi_s_sum",
        color=["celltype"],
        cmap="viridis",
        legend_loc=False,
        title="",
        ax=ax,
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_ssum_scvi_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_ssum_scvi_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )


# %% [markdown]
# #### Nuc-cyt umap scglue

# %%
adata.uns["celltype_colors"] = celltype_colors
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_nuc", n_jobs=8, xkey="Ms_nuc")
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_nuc", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_snuc_scglue_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_snuc_scglue_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )

# %%
adata.uns["celltype_colors"] = celltype_colors
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_cyt", n_jobs=8, xkey="Ms_cyt")
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_cyt", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_scyt_scglue_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_scyt_scglue_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )


# %%
adata.layers["Ms_sum"] = adata.layers["Ms_nuc"] + adata.layers["Ms_cyt"]
adata.layers["velocities_velovi_s_sum"] = (
    adata.layers["velocities_velovi_s_nuc"] + adata.layers["velocities_velovi_s_cyt"]
)
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_sum", n_jobs=2, xkey="Ms_sum")

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_sum", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_streams"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_ssum_scglue_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_ssum_scglue_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )

# %% [markdown]
# #### Single-cell umap

# %%
scv.tl.umap(adata_sc)
scv.pl.velocity_embedding_stream(
    adata_sc, vkey="velocities_velovi", color=["celltype"], cmap="viridis", legend_loc=False, title=""
)
plt.show()

# %% [markdown]
# #### Single-nucleus umap

# %%
scv.tl.umap(adata_sn)
scv.pl.velocity_embedding_stream(
    adata_sn, vkey="velocities_velovi", color=["celltype"], cmap="viridis", legend_loc=False, title=""
)
plt.show()

# %% [markdown]
# ## Now we want to investigate inferred velocities of different genes

# %%
plot_genes = ["Hells", "Top2a", "Sulf2"]
# get common cell names
sc_cells = list(set(adata_sc.obs_names).intersection(set(adata.obs_names)))
sn_cells = list(set(adata_sn.obs_names).intersection(set(adata.obs_names)))

# %% [markdown]
# #### 1. Compare the inferred velo_modes for the Nuc/Cyt model
# PLot phase portraits

# %%
# define velocities_velovi_s_nuc_u, velocities_velovi_s_cyt_u for plotting purposes
adata.layers["velocities_velovi_s_cyt_u"] = adata.layers["velocities_velovi_u_nuc"].copy()
adata.layers["velocities_velovi_s_nuc_u"] = adata.layers["velocities_velovi_u_nuc"].copy()
adata.layers["velocities_velovi_s_sum_u"] = adata.layers["velocities_velovi_u_nuc"].copy()

for gene in plot_genes:
    # Ms_nuc vs Mu_nuc
    adata.layers["Mu"] = adata.layers["Mu_nuc"]
    adata.layers["Ms"] = adata.layers["Ms_nuc"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_nuc",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_nuc", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_snuc_velo_snuc_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_cyt",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_nuc", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_scyt_velo_snuc_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_sum",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_nuc", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_ssum_velo_snuc_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    # Ms_cyt vs. Mu_nuc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    adata.layers["Mu"] = adata.layers["Mu_nuc"]
    adata.layers["Ms"] = adata.layers["Ms_cyt"]
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_nuc",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_snuc_velo_scyt_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_cyt",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_scyt_velo_scyt_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_sum",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Mu_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_ssum_velo_scyt_unuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    # Ms_cyt vs. Ms_nuc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    adata.layers["Mu"] = adata.layers["Ms_nuc"]
    adata.layers["Ms"] = adata.layers["Ms_cyt"]
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_nuc",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Ms_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_snuc_velo_scyt_snuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_cyt",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Ms_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_scyt_velo_scyt_snuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax = scv.pl.velocity_embedding(
        adata,
        vkey="velocities_velovi_s_sum",
        basis=gene,
        fontsize=16,
        frameon=False,
        color="celltype",
        legend_loc=None,
        show=False,
        title="",
        ax=ax,
    )
    scv.pl.plot_nuc_cyt_dynamics(adata, gene, "purple", ax, "Ms_cyt", "Ms_nuc")
    plt.show()
    fig.savefig(
        FIG_DIR / f"{gene}_ssum_velo_scyt_snuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
    )

# %% [markdown]
# ## same for scvi

# %%
# define velocities_velovi_s_nuc_u, velocities_velovi_s_cyt_u for plotting purposes
adata_scvi.layers["velocities_velovi_s_cyt_u"] = adata_scvi.layers["velocities_velovi_u_nuc"].copy()
adata_scvi.layers["velocities_velovi_s_nuc_u"] = adata_scvi.layers["velocities_velovi_u_nuc"].copy()
adata_scvi.layers["velocities_velovi_s_sum_u"] = adata_scvi.layers["velocities_velovi_u_nuc"].copy()

# %% [markdown]
# ## Compare inferred velocities between models

# %% [markdown]
# #### 1. Compare velocities of model trained on scRNA data and the Nuc/Cyt model

# %% [markdown]
# velocities_velovi_s_nuc

# %%
palette = celltype_colors
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sc-model velocity"] = adata_sc[sc_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sc_cells, gene].to_df("velocities_velovi_s_nuc")
        df["Celltype"] = adata[sc_cells, gene].obs.celltype
        clipy = (df["sc-model velocity"].min(), df["sc-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sc-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sc-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")

        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sc-model velocities",
            pearsonr(
                adata_sc[sc_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sc_cells, gene].to_df("velocities_velovi_s_nuc").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_nuc_{gene}_nuccyt_sc_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# velocities_velovi_s_cyt

# %%
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sc-model velocity"] = adata_sc[sc_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sc_cells, gene].to_df("velocities_velovi_s_cyt")
        df["Celltype"] = adata[sc_cells, gene].obs.celltype
        clipy = (df["sc-model velocity"].min(), df["sc-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sc-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sc-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")
        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sc-model velocities",
            pearsonr(
                adata_sc[sc_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sc_cells, gene].to_df("velocities_velovi_s_cyt").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_cyt_{gene}_nuccyt_sc_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %%
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sc-model velocity"] = adata_sc[sc_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sc_cells, gene].to_df("velocities_velovi_s_sum")
        df["Celltype"] = adata[sc_cells, gene].obs.celltype
        clipy = (df["sc-model velocity"].min(), df["sc-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sc-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sc-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")
        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sc-model velocities",
            pearsonr(
                adata_sc[sc_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sc_cells, gene].to_df("velocities_velovi_s_sum").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_sum_{gene}_nuccyt_sc_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# #### 2. Compare velocities of model trained on snRNA data and the Nuc/Cyt model

# %% [markdown]
# velocities_velovi_s_nuc

# %%
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sn-model velocity"] = adata_sn[sn_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sn_cells, gene].to_df("velocities_velovi_s_nuc")
        df["Celltype"] = adata[sn_cells, gene].obs.celltype
        clipy = (df["sn-model velocity"].min(), df["sn-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sn-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sn-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")
        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sn-model velocities",
            pearsonr(
                adata_sn[sn_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sn_cells, gene].to_df("velocities_velovi_s_nuc").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_nuc_{gene}_nuccyt_sn_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# velocities_velovi_s_cyt

# %%
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sn-model velocity"] = adata_sn[sn_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sn_cells, gene].to_df("velocities_velovi_s_cyt")
        df["Celltype"] = adata[sn_cells, gene].obs.celltype
        clipy = (df["sn-model velocity"].min(), df["sn-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sn-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sn-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")
        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sn-model velocities",
            pearsonr(
                adata_sn[sn_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sn_cells, gene].to_df("velocities_velovi_s_cyt").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_cyt_{gene}_nuccyt_sn_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# velocity s_sum

# %%
for gene in plot_genes:
    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.015
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.1]
        rect_histy = [left + width + spacing, bottom, 0.1, height]

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.axis("off")
        ax_histy.axis("off")

        df = pd.DataFrame()
        df["sn-model velocity"] = adata_sn[sn_cells, gene].to_df("velocities_velovi")
        df["Nuc/Cyt model velocity"] = adata[sn_cells, gene].to_df("velocities_velovi_s_sum")
        df["Celltype"] = adata[sn_cells, gene].obs.celltype
        clipy = (df["sn-model velocity"].min(), df["sn-model velocity"].max())
        clipx = (df["Nuc/Cyt model velocity"].min(), df["Nuc/Cyt model velocity"].max())
        sns.kdeplot(
            data=df,
            y="sn-model velocity",
            hue="Celltype",
            ax=ax_histy,
            legend=False,
            clip=clipy,
            palette=palette,
        )
        sns.kdeplot(
            data=df,
            x="Nuc/Cyt model velocity",
            hue="Celltype",
            ax=ax_histx,
            legend=False,
            clip=clipx,
            palette=palette,
        )
        sns.scatterplot(
            y="sn-model velocity",
            x="Nuc/Cyt model velocity",
            data=df,
            hue="Celltype",
            s=4,
            palette=palette,
            ax=ax,
            legend=False,
        )
        plt.title(f"{gene}")
        plt.show()
        print(
            f"Pearson correlation for {gene} of inferred Nuc/Cyt-model velocities and sn-model velocities",
            pearsonr(
                adata_sn[sn_cells, gene].to_df("velocities_velovi").squeeze(),
                adata[sn_cells, gene].to_df("velocities_velovi_s_sum").squeeze(),
            ),
            "\n",
        )
        if SAVE_FIGURES:
            path = FIG_DIR / "velocity_comparison"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path / f"velo_s_sum_{gene}_nuccyt_sn_e15.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )

# %% [markdown]
# ### Pearson correlation for all genes

# %% [markdown]
# ## Compare scVI, scglue based models and check which `velo_mode` yields the highest correlation

# %%
# common_genes_scvi = list(set(adata_sc.var_names).intersection(set(adata_sn.var_names)).intersection(set(adata_scvi.var_names)).intersection(set(adata_integrated.var_names)))
sc_cells_scvi = list(set(adata_sc.obs_names).intersection(set(adata_scvi.obs_names)))
sn_cells_scvi = list(set(adata_sn.obs_names).intersection(set(adata_scvi.obs_names)))
sc_cells = list(set(adata_sc.obs_names).intersection(set(adata.obs_names)))
sn_cells = list(set(adata_sn.obs_names).intersection(set(adata.obs_names)))

common_genes = list(
    set(adata_sc.var_names)
    .intersection(set(adata_sn.var_names))
    .intersection(set(adata.var_names))
    .intersection(set(adata_scvi.var_names))
)
sc_cells = list(set(adata_sc.obs_names).intersection(set(adata.obs_names)))
sn_cells = list(set(adata_sn.obs_names).intersection(set(adata.obs_names)))

# %% [markdown]
# ## Plot correlations

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_sc_nuc_cyt = []
pearson_coeffs_sn_nuc_cyt = []

pearson_coeffs_sc_nuc_cyt_scvi = []
pearson_coeffs_sn_nuc_cyt_scvi = []

for gene in common_genes:
    pearson_coeffs_sc_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sc_cells_scvi, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sc[sc_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sn_cells_scvi, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sn[sn_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sc_nuc_cyt.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )


pearson_df = pd.DataFrame()
pearson_df["Nuc/Cyt model vs. sc-model (scglue)"] = pearson_coeffs_sc_nuc_cyt
pearson_df["Nuc/Cyt model vs. sn-model (scglue)"] = pearson_coeffs_sn_nuc_cyt
pearson_df["Nuc/Cyt model vs. sc-model (scVI)"] = pearson_coeffs_sc_nuc_cyt_scvi
pearson_df["Nuc/Cyt model vs. sn-model (scVI)"] = pearson_coeffs_sn_nuc_cyt_scvi

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        data=pearson_df,
        color="silver",
        medianprops={"color": "black"},
    )
    ax.set_ylabel("Pearson correlation of inferred velocities")
    ax.set_ylim([-1, 1])
    plt.xticks(rotation=45, ha="right")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_s_cyt_pearson_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## snuc velo

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_sc_nuc_cyt = []
pearson_coeffs_sn_nuc_cyt = []

pearson_coeffs_sc_nuc_cyt_scvi = []
pearson_coeffs_sn_nuc_cyt_scvi = []

for gene in common_genes:
    pearson_coeffs_sc_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sc_cells_scvi, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sn_cells_scvi, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sc_nuc_cyt.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )


pearson_df = pd.DataFrame()
pearson_df["Nuc/Cyt model vs. sc-model (scglue)"] = pearson_coeffs_sc_nuc_cyt
pearson_df["Nuc/Cyt model vs. sn-model (scglue)"] = pearson_coeffs_sn_nuc_cyt
pearson_df["Nuc/Cyt model vs. sc-model (scVI)"] = pearson_coeffs_sc_nuc_cyt_scvi
pearson_df["Nuc/Cyt model vs. sn-model (scVI)"] = pearson_coeffs_sn_nuc_cyt_scvi

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        data=pearson_df,
        color="silver",
        medianprops={"color": "black"},
    )
    ax.set_ylabel("Pearson correlation of inferred velocities")
    ax.set_ylim([-1, 1])
    plt.xticks(rotation=45, ha="right")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_s_cyt_pearson_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## s sum velo

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_sc_nuc_cyt = []
pearson_coeffs_sn_nuc_cyt = []

pearson_coeffs_sc_nuc_cyt_scvi = []
pearson_coeffs_sn_nuc_cyt_scvi = []

for gene in common_genes:
    pearson_coeffs_sc_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sc_cells_scvi, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sc[sc_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt_scvi.append(
        pearsonr(
            adata_scvi[sn_cells_scvi, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sn[sn_cells_scvi, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sc_nuc_cyt.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

    pearson_coeffs_sn_nuc_cyt.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )


pearson_df = pd.DataFrame()
pearson_df["Nuc/Cyt model vs. sc-model (scglue)"] = pearson_coeffs_sc_nuc_cyt
pearson_df["Nuc/Cyt model vs. sn-model (scglue)"] = pearson_coeffs_sn_nuc_cyt
pearson_df["Nuc/Cyt model vs. sc-model (scVI)"] = pearson_coeffs_sc_nuc_cyt_scvi
pearson_df["Nuc/Cyt model vs. sn-model (scVI)"] = pearson_coeffs_sn_nuc_cyt_scvi

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        data=pearson_df,
        color="silver",
        medianprops={"color": "black"},
    )
    ax.set_ylabel("Pearson correlation of inferred velocities")
    ax.set_ylim([-1, 1])
    plt.xticks(rotation=45, ha="right")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_s_sum_pearson_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Finally compare scvi and scglue velocities

# %%
scvi_scglue_cells = list(set(adata_scvi.obs_names).intersection(set(adata.obs_names)))

common_genes = list(set(adata.var_names).intersection(set(adata_scvi.var_names)))


pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_scvi_scglue_velo_snuc = []
pearson_coeffs_scvi_scglue_velo_scyt = []
pearson_coeffs_scvi_scglue_velo_ssum = []

for gene in common_genes:
    pearson_coeffs_scvi_scglue_velo_snuc.append(
        pearsonr(
            adata_scvi[scvi_scglue_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata[scvi_scglue_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
        )[0]
    )

    pearson_coeffs_scvi_scglue_velo_scyt.append(
        pearsonr(
            adata_scvi[scvi_scglue_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata[scvi_scglue_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
        )[0]
    )

    pearson_coeffs_scvi_scglue_velo_ssum.append(
        pearsonr(
            adata_scvi[scvi_scglue_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata[scvi_scglue_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
        )[0]
    )


pearson_df = pd.DataFrame()
pearson_df["scglue vs. scvi snuc"] = pearson_coeffs_scvi_scglue_velo_snuc
pearson_df["scglue vs. scvi scyt"] = pearson_coeffs_scvi_scglue_velo_scyt
pearson_df["scglue vs. scvi ssum"] = pearson_coeffs_scvi_scglue_velo_ssum


# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        data=pearson_df,
        color="silver",
        medianprops={"color": "black"},
    )
    ax.set_ylim([-1, 1])
    ax.set_ylabel("Pearson correlation of inferred velocities")
    plt.xticks(rotation=45, ha="right")
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velocity_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_scglue_vs_scvi_correlation_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Final plot
# for each velo mode one plot

# %%
names = ["Nuc/Cyt model", "sc-model", "sn-model"]

palette = dict(zip(names, sns.color_palette("colorblind").as_hex()[:3]))

# %% [markdown]
# ##### scyt velo

# %%
sc_cells = list(set(adata_scvi.obs_names).intersection(set(adata.obs_names)).intersection(set(adata_sc.obs_names)))
sn_cells = list(set(adata_scvi.obs_names).intersection(set(adata.obs_names)).intersection(set(adata_sn.obs_names)))

common_genes_sc = list(
    set(adata.var_names).intersection(set(adata_scvi.var_names).intersection(set(adata_sc.var_names)))
)

common_genes_sn = list(
    set(adata.var_names).intersection(set(adata_scvi.var_names).intersection(set(adata_sn.var_names)))
)


pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_scglue_sc = []
pearson_coeffs_scglue_sn = []
pearson_coeffs_scvi_sc = []
pearson_coeffs_scvi_sn = []

for gene in common_genes_sc:
    pearson_coeffs_scglue_sc.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sc.append(
        pearsonr(
            adata_scvi[sc_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

for gene in common_genes_sn:
    pearson_coeffs_scglue_sn.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sn.append(
        pearsonr(
            adata_scvi[sn_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

pearson_df = pd.DataFrame()
pearson_df["Correlation"] = (
    pearson_coeffs_scglue_sc + pearson_coeffs_scglue_sn + pearson_coeffs_scvi_sc + pearson_coeffs_scvi_sn
)
pearson_df["Model"] = ["scglue"] * (len(pearson_coeffs_scglue_sc) + len(pearson_coeffs_scglue_sn)) + ["scvi"] * (
    len(pearson_coeffs_scvi_sc) + len(pearson_coeffs_scvi_sn)
)
pearson_df["comparison"] = (
    ["sc-model"] * len(pearson_coeffs_scglue_sc)
    + ["sn-model"] * len(pearson_coeffs_scglue_sn)
    + ["sc-model"] * len(pearson_coeffs_scvi_sc)
    + ["sn-model"] * len(pearson_coeffs_scvi_sn)
)


# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=pearson_df,
        x="Model",
        y="Correlation",
        hue="comparison",
        palette=palette,  # sns.color_palette("colorblind").as_hex()[:3],
    )
    y_min, y_max = ax.get_ylim()

    # compare scglue sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.2,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi vs scglue (for sc)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    ax.set_ylim([y_min, y_max + 0.5])

    # compare scvi vs scglue (for sn)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.20,
        right=1.2,
        significance=significance,
        lw=1,
        bracket_level=0.92,
        c="k",
        level=0,
    )

    ax.set_ylim([-1, 1.75])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    if SAVE_FIGURES:
        path = FIG_DIR / "velo_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_compare_scvi_scglue_scyt_e15.svg", format="svg", transparent=True, bbox_inches="tight")


# %% [markdown]
# #### snuc

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_scglue_sc = []
pearson_coeffs_scglue_sn = []
pearson_coeffs_scvi_sc = []
pearson_coeffs_scvi_sn = []

for gene in common_genes_sc:
    pearson_coeffs_scglue_sc.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sc.append(
        pearsonr(
            adata_scvi[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

for gene in common_genes_sn:
    pearson_coeffs_scglue_sn.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sn.append(
        pearsonr(
            adata_scvi[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

pearson_df = pd.DataFrame()
pearson_df["Correlation"] = (
    pearson_coeffs_scglue_sc + pearson_coeffs_scglue_sn + pearson_coeffs_scvi_sc + pearson_coeffs_scvi_sn
)
pearson_df["Model"] = ["scglue"] * (len(pearson_coeffs_scglue_sc) + len(pearson_coeffs_scglue_sn)) + ["scvi"] * (
    len(pearson_coeffs_scvi_sc) + len(pearson_coeffs_scvi_sn)
)
pearson_df["comparison"] = (
    ["sc-model"] * len(pearson_coeffs_scglue_sc)
    + ["sn-model"] * len(pearson_coeffs_scglue_sn)
    + ["sc-model"] * len(pearson_coeffs_scvi_sc)
    + ["sn-model"] * len(pearson_coeffs_scvi_sn)
)


# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=pearson_df,
        x="Model",
        y="Correlation",
        hue="comparison",
        palette=palette,  # sns.color_palette("colorblind").as_hex()[:3],
    )
    y_min, y_max = ax.get_ylim()

    # compare scglue sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.2,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi vs scglue (for sc)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    ax.set_ylim([y_min, y_max + 0.5])

    # compare scvi vs scglue (for sn)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.20,
        right=1.2,
        significance=significance,
        lw=1,
        bracket_level=0.92,
        c="k",
        level=0,
    )

    ax.set_ylim([-1, 1.75])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    if SAVE_FIGURES:
        path = FIG_DIR / "velo_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_compare_scvi_scglue_snuc_e15.svg", format="svg", transparent=True, bbox_inches="tight")


# %% [markdown]
# ##### ssum

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_scglue_sc = []
pearson_coeffs_scglue_sn = []
pearson_coeffs_scvi_sc = []
pearson_coeffs_scvi_sn = []

for gene in common_genes_sc:
    pearson_coeffs_scglue_sc.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sc.append(
        pearsonr(
            adata_scvi[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

for gene in common_genes_sn:
    pearson_coeffs_scglue_sn.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scvi_sn.append(
        pearsonr(
            adata_scvi[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )

pearson_df = pd.DataFrame()
pearson_df["Correlation"] = (
    pearson_coeffs_scglue_sc + pearson_coeffs_scglue_sn + pearson_coeffs_scvi_sc + pearson_coeffs_scvi_sn
)
pearson_df["Model"] = ["scglue"] * (len(pearson_coeffs_scglue_sc) + len(pearson_coeffs_scglue_sn)) + ["scvi"] * (
    len(pearson_coeffs_scvi_sc) + len(pearson_coeffs_scvi_sn)
)
pearson_df["comparison"] = (
    ["sc-model"] * len(pearson_coeffs_scglue_sc)
    + ["sn-model"] * len(pearson_coeffs_scglue_sn)
    + ["sc-model"] * len(pearson_coeffs_scvi_sc)
    + ["sn-model"] * len(pearson_coeffs_scvi_sn)
)


# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=pearson_df,
        x="Model",
        y="Correlation",
        hue="comparison",
        palette=palette,  # sns.color_palette("colorblind").as_hex()[:3],
    )
    y_min, y_max = ax.get_ylim()

    # compare scglue sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.2,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi sn vs sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison != "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare scvi vs scglue (for sc)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sc-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    ax.set_ylim([y_min, y_max + 0.5])

    # compare scvi vs scglue (for sn)
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.Model == "scvi") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.Model == "scglue") & (pearson_df.comparison == "sn-model") & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.20,
        right=1.2,
        significance=significance,
        lw=1,
        bracket_level=0.92,
        c="k",
        level=0,
    )

    ax.set_ylim([-1, 1.75])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    if SAVE_FIGURES:
        path = FIG_DIR / "velo_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_compare_scvi_scglue_ssum_e15.svg", format="svg", transparent=True, bbox_inches="tight")


# %% [markdown]
# ## Finally compare velo modes

# %%
pearson_dfs = pd.DataFrame(columns=["Pearson correlation"])
pearson_coeffs_scglue_sc_velo_scyt = []
pearson_coeffs_scglue_sc_velo_snuc = []
pearson_coeffs_scglue_sc_velo_ssum = []
pearson_coeffs_scglue_sn_velo_scyt = []
pearson_coeffs_scglue_sn_velo_snuc = []
pearson_coeffs_scglue_sn_velo_ssum = []

for gene in common_genes_sc:
    pearson_coeffs_scglue_sc_velo_scyt.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scglue_sc_velo_snuc.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scglue_sc_velo_ssum.append(
        pearsonr(
            adata[sc_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sc[sc_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )


for gene in common_genes_sn:
    pearson_coeffs_scglue_sn_velo_scyt.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_cyt"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scglue_sn_velo_snuc.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_nuc"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )
    pearson_coeffs_scglue_sn_velo_ssum.append(
        pearsonr(
            adata[sn_cells, gene].layers["velocities_velovi_s_sum"].squeeze(),
            adata_sn[sn_cells, gene].layers["velocities_velovi"].squeeze(),
        )[0]
    )


# %%
pearson_df = pd.DataFrame()
pearson_df["Correlation"] = (
    pearson_coeffs_scglue_sc_velo_scyt
    + pearson_coeffs_scglue_sc_velo_snuc
    + pearson_coeffs_scglue_sc_velo_ssum
    + pearson_coeffs_scglue_sn_velo_scyt
    + pearson_coeffs_scglue_sn_velo_snuc
    + pearson_coeffs_scglue_sn_velo_ssum
)

pearson_df["VeloMode"] = (
    ["scyt"] * len(pearson_coeffs_scglue_sc_velo_scyt)
    + ["snuc"] * len(pearson_coeffs_scglue_sc_velo_snuc)
    + ["ssum"] * len(pearson_coeffs_scglue_sc_velo_ssum)
    + ["scyt"] * len(pearson_coeffs_scglue_sn_velo_scyt)
    + ["snuc"] * len(pearson_coeffs_scglue_sn_velo_snuc)
    + ["ssum"] * len(pearson_coeffs_scglue_sn_velo_ssum)
)

pearson_df["comparison"] = ["sc-model"] * (3 * len(pearson_coeffs_scglue_sc_velo_snuc)) + ["sn-model"] * (
    3 * len(pearson_coeffs_scglue_sn)
)


# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=pearson_df,
        x="VeloMode",
        y="Correlation",
        hue="comparison",
        palette=palette,
        order=["ssum", "snuc", "scyt"],
    )
    y_min, y_max = ax.get_ylim()
    # compare ssum to snuc for sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.VeloMode == "ssum")
            & (pearson_df.comparison == "sc-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.VeloMode == "snuc")
            & (pearson_df.comparison == "sc-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.20,
        right=0.8,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )
    # compare ssum to snuc for sn
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.VeloMode == "ssum")
            & (pearson_df.comparison == "sn-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.VeloMode == "snuc")
            & (pearson_df.comparison == "sn-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.20,
        right=1.2,
        significance=significance,
        lw=1,
        bracket_level=1.1,
        c="k",
        level=0,
    )

    # compare snuc to scyt for sc
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.VeloMode == "snuc")
            & (pearson_df.comparison == "sc-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.VeloMode == "scyt")
            & (pearson_df.comparison == "sc-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.8,
        right=1.8,
        significance=significance,
        lw=1,
        bracket_level=1.1,
        c="k",
        level=0,
    )

    # compare snuc to scyt for sn
    ttest_res = ttest_ind(
        pearson_df.loc[
            (pearson_df.VeloMode == "snuc")
            & (pearson_df.comparison == "sn-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        pearson_df.loc[
            (pearson_df.VeloMode == "scyt")
            & (pearson_df.comparison == "sn-model")
            & ~(pearson_df.Correlation.isnull()),
            "Correlation",
        ].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1.2,
        right=2.2,
        significance=significance,
        lw=1,
        bracket_level=1.1,
        c="k",
        level=0,
    )

    ax.set_ylim([-1, 2])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    if SAVE_FIGURES:
        path = FIG_DIR / "velo_comparison"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "compare_velo_modes_scglue_e15.svg", format="svg", transparent=True, bbox_inches="tight")
