# %% [markdown]
# # Uncertainty analysis - Pancreas

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys

import cellrank as cr
import mplscience
import torch
from velovi import VELOVI

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
from scvelo.core import l2_norm
from scvelo.inference import fit_velovi
from scvelo.inference._model import _compute_directional_statistics_tensor
from scvelo.preprocessing.neighbors import get_neighs
from scvelo.tools.utils import get_indices

sys.path.append("../..")
from paths import FIG_DIR, PROJECT_DIR  # isort: skip  # noqa: E402


# %% [markdown]
# ## Function definitions


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
    # scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=8)
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


# %% [markdown]
# ## General Settings

# %%
SAVE_FIGURES = True

sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

SN_PROCESSED_DIR = PROJECT_DIR / "pancreas_multiome" / "processed"
SC_PROCESSED_DIR = PROJECT_DIR / "pancreas_sc" / "processed"


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
# ## Read and Preprocess data

# %%
adata = sc.read(PROJECT_DIR / "pancreas_sc_multiome" / "sn_sc_rna_scglue_e15.5.h5ad")

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

scv.tl.umap(adata_sc)
scv.tl.umap(adata_sn)

# %% [markdown]
# ## Fit model

# %%
vae, adata = fit_velovi(
    adata, max_epochs=500, unspliced_layer_nuc="Mu_nuc", spliced_layer_nuc="Ms_nuc", spliced_layer_cyt="Ms_cyt", lr=5e-3
)

# %% [markdown]
# ## Fit sc and sn models

# %%
vae_sn, adata_sn = fit_velovi_(adata_sn)

# %%
vae_sc, adata_sc = fit_velovi_(adata_sc)

# %% [markdown]
# ## Velocity sampling

# %%
velocities = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_cyt")
adata.layers["velocities_velovi_s_cyt"] = velocities
velocities = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_nuc")
adata.layers["velocities_velovi_s_nuc"] = velocities
velocities = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced_sum")
adata.layers["velocities_velovi_s_sum"] = velocities


velocities = vae_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced")
adata_sc.layers["velocities_velovi"] = velocities
velocities = vae_sc.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")
adata_sc.layers["velocities_velovi_u"] = velocities


velocities = vae_sn.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="spliced")
adata_sn.layers["velocities_velovi"] = velocities
velocities = vae_sn.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")
adata_sn.layers["velocities_velovi_u"] = velocities

# %%
adata.layers["Ms_sum"] = adata.layers["Ms_nuc"] + adata.layers["Ms_cyt"]
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_cyt", n_jobs=3, xkey="Ms_cyt")
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_nuc", n_jobs=3, xkey="Ms_nuc")
scv.tl.velocity_graph(adata, vkey="velocities_velovi_s_sum", n_jobs=3, xkey="Ms_sum")
scv.tl.velocity_graph(adata_sc, vkey="velocities_velovi", n_jobs=3, xkey="Ms")
scv.tl.velocity_graph(adata_sn, vkey="velocities_velovi", n_jobs=3, xkey="Ms")

# %% [markdown]
# ## Intrinsic Uncertainty

# %%
# add cell type color here again
adata.uns["celltype_colors"] = celltype_colors
adata_sc.uns["celltype_colors"] = celltype_colors
adata_sn.uns["celltype_colors"] = celltype_colors

# %% [markdown]
# ## Spliced sum velo embedding and intrinsic uncertainty

# %%
adata.uns["celltype_colors"] = celltype_colors
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_sum", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_s_sum.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(path / "velo_embedding_s_sum_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight")

# %%
# Intrinsic
udf, _ = vae.get_directional_uncertainty(adata, velo_mode="spliced_sum", n_samples=50, n_jobs=3)

for c in udf.columns:
    adata.obs[c] = np.log10(udf[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance", perc=[5, 95], title="", cmap="Greys", ax=ax)

if SAVE_FIGURES:
    path = FIG_DIR / "uncertainty"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / "directional_cosine_sim_variance_initrinsic_ssum_e15.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Spliced nucleic velo embedding and intrinsic uncertainty

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_nuc", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_snuc_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(path / "velo_embedding_snuc_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight")

# %%
# Intrinsic
udf, _ = vae.get_directional_uncertainty(adata, velo_mode="spliced_nuc", n_samples=50, n_jobs=3)

for c in udf.columns:
    adata.obs[c] = np.log10(udf[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance", perc=[5, 95], title="", cmap="Greys", ax=ax)

if SAVE_FIGURES:
    path = FIG_DIR / "uncertainty"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / "directional_cosine_sim_variance_initrinsic_snuc_e15.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Spliced nucleic velo embedding and intrinsic uncertainty

# %%
adata.uns["celltype_colors"] = celltype_colors
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata, vkey="velocities_velovi_s_cyt", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_s_cyt_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(path / "velo_embedding_s_cyt_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight")

# %%
# Intrinsic
udf, _ = vae.get_directional_uncertainty(adata, velo_mode="spliced_cyt", n_samples=50, n_jobs=3)

for c in udf.columns:
    adata.obs[c] = np.log10(udf[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance", perc=[5, 95], title="", cmap="Greys", ax=ax)

if SAVE_FIGURES:
    path = FIG_DIR / "uncertainty"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / "directional_cosine_sim_variance_initrinsic_scyt_e15.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Single-cell uncertainties

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sc, vkey="velocities_velovi", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "velo_embedding_sc_e15_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight"
        )

# %%
# Intrinsic
udf, _ = vae_sc.get_directional_uncertainty(adata_sc, n_samples=50)

for c in udf.columns:
    adata_sc.obs[c] = np.log10(udf[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata_sc, color="directional_cosine_sim_variance", perc=[5, 95], title="", cmap="Greys", ax=ax)

if SAVE_FIGURES:
    path = FIG_DIR / "uncertainty"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / "directional_cosine_sim_variance_initrinsic_sc_e15.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Single-nucleus uncertainties

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sn, vkey="velocities_velovi", color=["celltype"], cmap="viridis", legend_loc=False, title="", ax=ax
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_embedding_sn_e15.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(path / "velo_embedding_sn_e15.png", format="png", dpi=500, transparent=True, bbox_inches="tight")

# %%
# Intrinsic
udf, _ = vae_sn.get_directional_uncertainty(adata_sn, n_samples=50)

for c in udf.columns:
    adata_sn.obs[c] = np.log10(udf[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata_sn, color="directional_cosine_sim_variance", perc=[5, 95], title="", cmap="Greys", ax=ax)

if SAVE_FIGURES:
    path = FIG_DIR / "uncertainty"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / "directional_cosine_sim_variance_initrinsic_sn_e15.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Extrinsic uncertainties

# %% [markdown]
# #### s_sum

# %%
extrapolated_cells_list = []
for i in range(25):
    vkey = f"velocities_velovi_{i}"
    v = vae.get_velocity(n_samples=1, velo_statistic="mean", velo_mode="spliced_sum")
    adata.layers[vkey] = v
    scv.tl.velocity_graph(adata, vkey=vkey, xkey="Ms_sum", sqrt_transform=False, approx=True)
    t_mat = scv.utils.get_transition_matrix(adata, vkey=vkey, self_transitions=True, use_negative_cosines=True)
    extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms_sum"])
    extrapolated_cells_list.append(extrapolated_cells)
extrapolated_cells = np.stack(extrapolated_cells_list)

# %%
df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=3, n_cells=adata.n_obs)

for c in df.columns:
    adata.obs[c + "_extrinsic"] = np.log10(df[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance_extrinsic", perc=[10, 90], cmap="viridis", ax=ax)

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "directional_cosine_sim_variance_extrinsic_ssum_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## scyt

# %%
extrapolated_cells_list = []
for i in range(25):
    vkey = f"velocities_velovi_{i}"
    v = vae.get_velocity(n_samples=1, velo_statistic="mean", velo_mode="spliced_cyt")
    adata.layers[vkey] = v
    scv.tl.velocity_graph(adata, vkey=vkey, xkey="Ms_cyt", sqrt_transform=False, approx=True)
    t_mat = scv.utils.get_transition_matrix(adata, vkey=vkey, self_transitions=True, use_negative_cosines=True)
    extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms_cyt"])
    extrapolated_cells_list.append(extrapolated_cells)
extrapolated_cells = np.stack(extrapolated_cells_list)

# %%
df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=4, n_cells=adata.n_obs)

for c in df.columns:
    adata.obs[c + "_extrinsic"] = np.log10(df[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance_extrinsic", perc=[10, 90], cmap="viridis", ax=ax)

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "directional_cosine_sim_variance_extrinsic_scyt_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## s_nuc

# %%
extrapolated_cells_list = []
for i in range(25):
    vkey = f"velocities_velovi_{i}"
    v = vae.get_velocity(n_samples=1, velo_statistic="mean", velo_mode="spliced_nuc")
    adata.layers[vkey] = v
    scv.tl.velocity_graph(adata, vkey=vkey, xkey="Ms_nuc", sqrt_transform=False, approx=True)
    t_mat = scv.utils.get_transition_matrix(adata, vkey=vkey, self_transitions=True, use_negative_cosines=True)
    extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms_nuc"])
    extrapolated_cells_list.append(extrapolated_cells)
extrapolated_cells = np.stack(extrapolated_cells_list)

# %%
df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=4, n_cells=adata.n_obs)

for c in df.columns:
    adata.obs[c + "_extrinsic"] = np.log10(df[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata, color="directional_cosine_sim_variance_extrinsic", perc=[10, 90], cmap="viridis", ax=ax)

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "directional_cosine_sim_variance_extrinsic_snuc_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Single-cell extrinsic uncertainty

# %%
# same for sc
extrapolated_cells_list = []
for i in range(25):
    vkey = f"velocities_velovi_{i}"
    v = vae_sc.get_velocity(n_samples=1, velo_statistic="mean")
    adata_sc.layers[vkey] = v
    scv.tl.velocity_graph(adata_sc, vkey=vkey, sqrt_transform=False, approx=True)
    t_mat = scv.utils.get_transition_matrix(adata_sc, vkey=vkey, self_transitions=True, use_negative_cosines=True)
    extrapolated_cells = np.asarray(t_mat @ adata_sc.layers["Ms"])
    extrapolated_cells_list.append(extrapolated_cells)
extrapolated_cells = np.stack(extrapolated_cells_list)

df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=2, n_cells=adata_sc.n_obs)

for c in df.columns:
    adata_sc.obs[c + "_extrinsic"] = np.log10(df[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata_sc, color="directional_cosine_sim_variance_extrinsic", perc=[10, 90], cmap="viridis", ax=ax)

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "directional_cosine_sim_variance_extrinsic_sc_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Single-nucleus extrinsic uncertainty

# %%
# same for sn
extrapolated_cells_list = []
for i in range(25):
    vkey = f"velocities_velovi_{i}"
    v = vae_sn.get_velocity(n_samples=1, velo_statistic="mean")
    adata_sn.layers[vkey] = v
    scv.tl.velocity_graph(adata_sn, vkey=vkey, sqrt_transform=False, approx=True)
    t_mat = scv.utils.get_transition_matrix(adata_sn, vkey=vkey, self_transitions=True, use_negative_cosines=True)
    extrapolated_cells = np.asarray(t_mat @ adata_sn.layers["Ms"])
    extrapolated_cells_list.append(extrapolated_cells)
extrapolated_cells = np.stack(extrapolated_cells_list)

df, _ = _compute_directional_statistics_tensor(extrapolated_cells, n_jobs=2, n_cells=adata_sn.n_obs)

for c in df.columns:
    adata_sn.obs[c + "_extrinsic"] = np.log10(df[c].values)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.umap(adata_sn, color="directional_cosine_sim_variance_extrinsic", perc=[10, 90], cmap="viridis", ax=ax)

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "directional_cosine_sim_variance_extrinsic_sn_e15.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Compare scale of uncertainties

# %%
uncertainty_df = pd.DataFrame()
nuc_cyt_in = adata.obs["directional_cosine_sim_variance"].tolist()
sc_in = adata_sc.obs["directional_cosine_sim_variance"].tolist()
sn_in = adata_sn.obs["directional_cosine_sim_variance"].tolist()

nuc_cyt_ex = adata.obs["directional_cosine_sim_variance_extrinsic"].tolist()
sc_ex = adata_sc.obs["directional_cosine_sim_variance_extrinsic"].tolist()
sn_ex = adata_sn.obs["directional_cosine_sim_variance_extrinsic"].tolist()


uncertainty_df["uncertainty_intrinsic"] = nuc_cyt_in + sc_in + sn_in
uncertainty_df["uncertainty_extrinsic"] = nuc_cyt_ex + sc_ex + sn_ex
uncertainty_df["log_uncertainty_extrinsic"] = -np.log10(np.abs(uncertainty_df["uncertainty_extrinsic"]))
uncertainty_df["Model"] = ["Nuc/Cyt model"] * len(nuc_cyt_in) + ["sc-model"] * len(sc_in) + ["sn-model"] * len(sn_in)

palette = dict(zip(["Nuc/Cyt model", "sc-model", "sn-model"], sns.color_palette("colorblind").as_hex()[:3]))

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(data=uncertainty_df, x="Model", y="uncertainty_intrinsic", palette=palette, ax=ax)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.7])

    # sc vs nuc cyt
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "Nuc/Cyt model", "uncertainty_intrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sc-model", "uncertainty_intrinsic"],
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.3,
        c="k",
        level=0,
    )

    # sc vs sn
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "sc-model", "uncertainty_intrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sn-model", "uncertainty_intrinsic"],
        equal_var=False,
        alternative="two-sided",
    )

    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=1.2,
        c="k",
        level=0,
    )

    # nuc cyt vs sn
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "Nuc/Cyt model", "uncertainty_intrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sn-model", "uncertainty_intrinsic"],
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=1.1,
        c="k",
        level=0,
    )
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        if not os.path.exists(path):
            path.mkdir()
        fig.savefig(path / "intrinsic_uncertainty_compare_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=uncertainty_df, x="Model", y="uncertainty_extrinsic", palette=palette, ax=ax)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 3])

    # sc vs nuc cyt
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "Nuc/Cyt model", "uncertainty_extrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sc-model", "uncertainty_extrinsic"],
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=0.2,
        c="k",
        level=0,
    )

    # sc vs sn
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "sc-model", "uncertainty_extrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sn-model", "uncertainty_extrinsic"],
        equal_var=False,
        alternative="two-sided",
    )

    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=1,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=0.4,
        c="k",
        level=0,
    )

    # nuc cyt vs sn
    ttest_res = ttest_ind(
        uncertainty_df.loc[uncertainty_df.Model == "Nuc/Cyt model", "uncertainty_extrinsic"],
        uncertainty_df.loc[uncertainty_df.Model == "sn-model", "uncertainty_extrinsic"],
        equal_var=False,
        alternative="two-sided",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=0.7,
        c="k",
        level=0,
    )
    plt.show()

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        if not os.path.exists(path):
            path.mkdir()
        fig.savefig(path / "extrinsic_uncertainty_compare_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Calculate Pseudotimes and plot against fate probabilities colored by uncertainties

# %% [markdown]
# #### 1. compute velocity kernel induced cell-cell transition matrix

# %%
vk = cr.kernels.VelocityKernel(adata, vkey="velocities_velovi_s_sum", xkey="Ms_sum")
vk.compute_transition_matrix()

vk.plot_projection(color="celltype", legend_loc="right margin")

vk.write_to_adata()

# %%
# vk = cr.kernels.VelocityKernel(adata, vkey="velocities_velovi_s_nuc", xkey="Ms_nuc")
# vk.compute_transition_matrix()

# vk.plot_projection(color="celltype", legend_loc="right margin")

# vk.write_to_adata()

# %%
vk = cr.kernels.VelocityKernel(adata_sc, vkey="velocities_velovi", xkey="Ms")
vk.compute_transition_matrix()

vk.plot_projection(color="celltype", legend_loc="right margin")

vk.write_to_adata()

# %%
vk = cr.kernels.VelocityKernel(adata_sn, vkey="velocities_velovi", xkey="Ms")
vk.compute_transition_matrix()

vk.plot_projection(color="celltype", legend_loc="right margin")

vk.write_to_adata()

# %% [markdown]
# ##### Follow steps from cellranks tutorial

# %%
# single-cell
vk = cr.kernels.VelocityKernel.from_adata(adata_sc, key="T_fwd")

g = cr.estimators.GPCCA(vk)
print(g)

g.fit(cluster_key="celltype", n_states=[4, 12])

# g.set_terminal_states(states=["Alpha", "Beta"])
g.set_terminal_states(states=["Delta", "Beta"])
g.plot_macrostates(which="terminal", legend_loc="right", size=100)

g.compute_fate_probabilities()
g.plot_fate_probabilities(same_plot=False)

# %%
g.plot_macrostates(which="all", discrete=True, legend_loc="right", s=100)

# %%
# calculate pseudotime
scv.tl.velocity_pseudotime(adata_sc, vkey="velocities_velovi")

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sc,
        vkey="velocities_velovi",
        color="velocities_velovi_pseudotime",
        color_map="gnuplot",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudtime_umap_velo_stream_sc_e15.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )
        fig.savefig(path / "pseudtime_umap_velo_stream_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["velocities_velovi_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="celltype",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sc_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["velocities_velovi_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance",
        perc=[5, 95],
        cmap="Greys",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sc_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["velocities_velovi_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance_extrinsic",
        perc=[10, 90],
        cmap="viridis",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sc_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %%
adata_sc.uns["iroot"] = np.flatnonzero(adata_sc.obs["celltype"] == "Ductal")[0]
sc.tl.dpt(adata_sc)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sc,
        vkey="velocities_velovi",
        color="dpt_pseudotime",
        color_map="gnuplot",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudtime_umap_velo_stream_sc_dpt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )
        fig.savefig(
            path / "pseudtime_umap_velo_stream_sc_dpt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )

# %%
## now for dpt pseudotime
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["dpt_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="celltype",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sc_dpt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sc_dpt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["dpt_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance",
        perc=[5, 95],
        cmap="Greys",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sc_dpt.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sc_dpt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sc,
        x=adata_sc.obs["dpt_pseudotime"],
        y=adata_sc.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance_extrinsic",
        perc=[10, 90],
        cmap="viridis",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sc_dpt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sc_dpt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ### Same for single-nucleus

# %%
# single-nucleus
vk = cr.kernels.VelocityKernel.from_adata(adata_sn, key="T_fwd")

g = cr.estimators.GPCCA(vk)
print(g)

g.fit(cluster_key="celltype", n_states=[4, 12])

# does not infer epsilon, delta as in tutorial
# g.set_terminal_states(states=["Alpha", "Beta"])
g.set_terminal_states(states=["Epsilon", "Beta"])
g.plot_macrostates(which="terminal", legend_loc="right", size=100)

g.compute_fate_probabilities()
g.plot_fate_probabilities(same_plot=False)

# %%
# calculate pseudotime
scv.tl.velocity_pseudotime(adata_sn, vkey="velocities_velovi")

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sn,
        vkey="velocities_velovi",
        color="velocities_velovi_pseudotime",
        color_map="gnuplot",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudtime_umap_velo_stream_sn.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )
        fig.savefig(path / "pseudtime_umap_velo_stream_sn.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
adata_sn.uns["iroot"] = np.flatnonzero(adata_sn.obs["celltype"] == "Ductal")[0]
sc.tl.dpt(adata_sn)

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata_sn,
        vkey="velocities_velovi",
        color="dpt_pseudotime",
        color_map="gnuplot",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudtime_umap_velo_stream_sn_dpt.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )
        fig.savefig(path / "pseudtime_umap_velo_stream_sn_dpt.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["velocities_velovi_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="celltype",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "pseudotime_fate_prob_celltype_sn.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sn.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["velocities_velovi_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance",
        perc=[5, 95],
        cmap="Greys",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "pseudotime_fate_prob_intrinsic_sn.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sn.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["velocities_velovi_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance_extrinsic",
        perc=[10, 90],
        cmap="viridis",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "pseudotime_fate_prob_extrinsic_sn.svg", format="svg", transparent=True, bbox_inches="tight")
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sn.png", format="png", dpi=700, transparent=True, bbox_inches="tight"
        )

# %%
## now for dpt pseudotime
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["dpt_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="celltype",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sn_dpt.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_sn_dpt.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["dpt_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance",
        perc=[5, 95],
        cmap="Greys",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sn_dpt.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_sn_dpt.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata_sn,
        x=adata_sn.obs["dpt_pseudotime"],
        y=adata_sn.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance_extrinsic",
        perc=[10, 90],
        cmap="viridis",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sn_dpt.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_sn_dpt.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Nuc/Cyt model

# %%
# Nuccyt
vk = cr.kernels.VelocityKernel.from_adata(adata, key="T_fwd")

g = cr.estimators.GPCCA(vk)
print(g)

g.fit(cluster_key="celltype", n_states=10)

# %%
# does not infer epsilon, delta as in tutorial
g.set_terminal_states(states=["Epsilon", "Alpha", "Beta", "Delta"])
g.plot_macrostates(which="terminal", legend_loc="right", size=100)

g.compute_fate_probabilities()
g.plot_fate_probabilities(same_plot=False)

# %%
# calculate pseudotime
adata.layers["Ms"] = adata.layers["Ms_sum"]
scv.tl.velocity_pseudotime(adata, vkey="velocities_velovi_s_sum")

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(
        adata,
        vkey="velocities_velovi_s_sum",
        color="velocities_velovi_s_sum_pseudotime",
        color_map="gnuplot",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )
    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudtime_umap_velo_stream_nuccyt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )
        fig.savefig(
            path / "pseudtime_umap_velo_stream_nuccyt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )

# %%
## now for dpt pseudotime
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata,
        x=adata.obs["velocities_velovi_s_sum_pseudotime"],
        y=adata.obs["term_states_fwd_probs"],
        color="celltype",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_nuccyt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_celltype_nuccyt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata,
        x=adata.obs["velocities_velovi_s_sum_pseudotime"],
        y=adata.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance",
        perc=[5, 95],
        cmap="Greys",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_nuccyt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_intrinsic_nuccyt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    scv.pl.scatter(
        adata,
        x=adata.obs["velocities_velovi_s_sum_pseudotime"],
        y=adata.obs["term_states_fwd_probs"],
        color="directional_cosine_sim_variance_extrinsic",
        perc=[10, 90],
        cmap="viridis",
        ax=ax,
        title="",
        legend_loc=False,
        colorbar=False,
    )

    if SAVE_FIGURES:
        path = FIG_DIR / "uncertainty"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_nuccyt_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )
        fig.savefig(
            path / "pseudotime_fate_prob_extrinsic_nuccyt_e15.png",
            format="png",
            dpi=700,
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ## Velocity confidence
# Calculate velo confidence of single-cell and nucleus velocities (single-modal models) by embedding into batch-corrected latent space and restrict neighborhood to neighbors of the respective other modality

# %%
# get common cell names
sc_cells = list(set(adata_sc.obs_names).intersection(set(adata.obs_names)))
sn_cells = list(set(adata_sn.obs_names).intersection(set(adata.obs_names)))

# find common genes
common_genes = list(set(adata_sc.var_names).intersection(set(adata_sn.var_names)).intersection(set(adata.var_names)))
velos_sn = adata_sn[sn_cells, common_genes].layers["velocities_velovi"]
velos_sc = adata_sc[sc_cells, common_genes].layers["velocities_velovi"]
velos_sn_sc = np.concatenate([velos_sn, velos_sc], axis=0)

adata_sn_sc = adata[sn_cells + sc_cells, common_genes].copy()
adata_sn_sc.layers["velocity_sn_sc"] = velos_sn_sc

vkey = "velocity_sn_sc"
V = np.array(adata_sn_sc.layers[vkey])

tmp_filter = np.invert(np.isnan(np.sum(V, axis=0)))
if f"{vkey}_genes" in adata_sn_sc.var.keys():
    tmp_filter &= np.array(adata_sn_sc.var[f"{vkey}_genes"], dtype=bool)
if "spearmans_score" in adata_sn_sc.var.keys():
    tmp_filter &= adata_sn_sc.var["spearmans_score"].values > 0.1

V = V[:, tmp_filter]

V -= V.mean(1)[:, None]
V_norm = l2_norm(V, axis=1)
R = np.zeros(adata_sn_sc.n_obs)

ad_obs = adata_sn_sc.obs.reset_index()
sc_index = ad_obs[ad_obs.protocol == "scRNA-seq"].index
sn_index = ad_obs[ad_obs.protocol != "scRNA-seq"].index

indices = get_indices(dist=get_neighs(adata_sn_sc, "distances"))[0]
for i in range(adata_sn_sc.n_obs):
    if adata_sn_sc.obs.protocol[i] == "scRNA-seq":
        Vi_neighs = V[list(set(indices[i]).intersection(sn_index))]
        Vi_neighs -= Vi_neighs.mean(1)[:, None]
        R[i] = np.mean(np.einsum("ij, j", Vi_neighs, V[i]) / (l2_norm(Vi_neighs, axis=1) * V_norm[i])[None, :])
    else:
        Vi_neighs = V[list(set(indices[i]).intersection(sc_index))]
        Vi_neighs -= Vi_neighs.mean(1)[:, None]
        R[i] = np.mean(np.einsum("ij, j", Vi_neighs, V[i]) / (l2_norm(Vi_neighs, axis=1) * V_norm[i])[None, :])

adata_sn_sc.obs[f"{vkey}_length"] = V_norm.round(2)
adata_sn_sc.obs[f"{vkey}_confidence"] = np.clip(R, 0, None)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata_sn_sc, c="velocity_sn_sc_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_confidence_sn_sc_neighbors_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )

# %% [markdown]
# ## Single-cell vs. nucleus confidence

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata_sn_sc, c="velocity_sn_sc_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path / "velo_confidence_sn_sc_neighbors_e15.svg", format="svg", transparent=True, bbox_inches="tight"
        )

# %% [markdown]
# ## Now for each model separatley

# %%
scv.tl.velocity_confidence(adata_sc, vkey="velocities_velovi")
scv.tl.velocity_confidence(adata_sn, vkey="velocities_velovi")

adata.layers["Ms"] = adata.layers["Ms_nuc"]
scv.tl.velocity_confidence(adata, vkey="velocities_velovi_s_nuc")
adata.layers["Ms"] = adata.layers["Ms_sum"]
scv.tl.velocity_confidence(adata, vkey="velocities_velovi_s_sum")

# %% [markdown]
# ## Nucleus cytosol model
# s_nuc velo

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, c="velocities_velovi_s_nuc_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_confidence_nuc_cyt_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# S_sum velo

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, c="velocities_velovi_s_sum_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_confidence_nuc_cyt_ssum_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Single-nucleus confidence

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata_sn, c="velocities_velovi_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_confidence_sn_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Single-cell confidence

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata_sc, c="velocities_velovi_confidence", cmap="coolwarm", perc=[5, 95], ax=ax, title="")
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_confidence_sc_e15.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Plot violin plot of all models and velo confidence distribution

# %%
dfs = []


adata.layers["Ms"] = adata.layers["Ms_nuc"]
g_df = compute_confidence(adata, "velocities_velovi_s_nuc")
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model (s_nuc velo)"
dfs.append(g_df)

adata.layers["Ms"] = adata.layers["Ms_cyt"]
g_df = compute_confidence(adata, "velocities_velovi_s_cyt")
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model (s_cyt velo)"
dfs.append(g_df)

adata.layers["Ms"] = adata.layers["Ms_sum"]
g_df = compute_confidence(adata, "velocities_velovi_s_sum")
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model (s_sum velo)"
dfs.append(g_df)

g_df = compute_confidence(adata_sc)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sc-model"
dfs.append(g_df)

g_df = compute_confidence(adata_sn)
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "sn-model"
dfs.append(g_df)

g_df = pd.DataFrame()
g_df["Velocity confidence"] = adata_sn_sc.obs["velocity_sn_sc_confidence"].to_numpy().ravel()
g_df["Dataset"] = "Pancreas E14.5"
g_df["Method"] = "Nuc/Cyt model sc neighbors vs sn neigbors"
dfs.append(g_df)


conf_df = pd.concat(dfs, axis=0)

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=conf_df,
        ax=ax,
        orient="v",
        x="Dataset",
        y="Velocity confidence",
        hue="Method",
        palette=sns.color_palette("colorblind").as_hex()[:6],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # compare snuc to scyt for sn
    ttest_res = ttest_ind(
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model (s_nuc velo)"), "Velocity confidence"].values,
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model (s_cyt velo)"), "Velocity confidence"].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.33,
        right=-0.2,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )
    # compare scyt ssum
    ttest_res = ttest_ind(
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model (s_cyt velo)"), "Velocity confidence"].values,
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model (s_sum velo)"), "Velocity confidence"].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.2,
        right=-0.07,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )
    # compare ssum sc
    ttest_res = ttest_ind(
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model (s_sum velo)"), "Velocity confidence"].values,
        conf_df.loc[(conf_df.Method == "sc-model"), "Velocity confidence"].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=-0.07,
        right=0.07,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )
    # compare sn sc
    ttest_res = ttest_ind(
        conf_df.loc[(conf_df.Method == "sc-model"), "Velocity confidence"].values,
        conf_df.loc[(conf_df.Method == "sn-model"), "Velocity confidence"].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.07,
        right=0.21,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )

    # compare sn to restricted
    ttest_res = ttest_ind(
        conf_df.loc[(conf_df.Method == "sn-model"), "Velocity confidence"].values,
        conf_df.loc[(conf_df.Method == "Nuc/Cyt model sc neighbors vs sn neigbors"), "Velocity confidence"].values,
        equal_var=False,
        alternative="greater",
    )
    significance = _get_significance(ttest_res.pvalue)
    _add_significance(
        ax=ax,
        left=0.21,
        right=0.35,
        significance=significance,
        lw=1,
        bracket_level=1,
        c="k",
        level=0,
    )
    ax.set_ylim([0, 1.3])
    plt.show()
    if SAVE_FIGURES:
        path = FIG_DIR / "velo_confidence"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "velo_confidence_all_violin_e15.svg", format="svg", transparent=True, bbox_inches="tight")
