# %% [markdown]
# # Data Preprocessing
#
# This notebook reads in Pancreas scRNA-seq data, splits the data by day E14.5 and E15.5 and removes redundant information
#
# **Requires:**
# - `/vol/storage/data/pancreas_sc/pancreas_2019.h5ad`
#
# **Output:**
# - `/vol/storage/data/pancreas_sc/processed/gex_e14.5.h5ad`
# - `/vol/storage/data/pancreas_sc/processed/gex_e15.5.h5ad`

# %% [markdown]
# ## Library imports

# %%
import sys

import scanpy as sc

sys.path.append("../../")
from paths import DATA_DIR, FIG_DIR, PROJECT_DIR  # isort: skip  # noqa: E402,F401

# %% [markdown]
# ## General settings

# %%
CELLTYPES_TO_KEEP = [
    "Alpha",
    "Beta",
    "Delta",
    "Ductal",
    "Epsilon",
    "Pre-endocrine",
    "Ngn3 high EP",
    "Ngn3 low EP",
]

SC_RAW_DIR = PROJECT_DIR / "pancreas_sc" / "raw"
SC_PROCESSED_DIR = PROJECT_DIR / "pancreas_sc" / "processed"

# If Processed folder doesnt exist, create it first
SC_PROCESSED_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
# Read scRNA data
adata_rna = sc.read(SC_RAW_DIR / "pancreas_2019.h5ad")
adata_rna

# %% [markdown]
# ## Data processing and saving
#

# %% [markdown]
# ### E14.5

# %%
adata = adata_rna[adata_rna.obs["day"].isin(["14.5"]), :]
adata.obs = adata.obs.loc[:, ["day", "celltype"]].replace(
    {
        "14.5": "E14.5",
        "Ngn3 High early": "Ngn3 high EP",
        "Ngn3 High late": "Ngn3 high EP",
        "Fev+ Pyy": "Eps/Delta progenitors",
        "Fev+ Epsilon": "Eps/Delta progenitors",
    }
)
adata.obs["celltype_fine"] = adata.obs["celltype"].copy()
adata.obs["celltype"].replace(
    {
        "Fev+ Alpha": "Pre-endocrine",
        "Fev+ Beta": "Pre-endocrine",
        "Fev+ Delta": "Pre-endocrine",
        "Eps/Delta progenitors": "Pre-endocrine",
    },
    inplace=True,
)
adata.obs["protocol"] = "scRNA-seq"
adata.obs = adata.obs.astype({"celltype": "category", "celltype_fine": "category"})

adata = adata[adata.obs["celltype"].isin(CELLTYPES_TO_KEEP), :].copy()

adata.var = adata.var.loc[:, []]
adata.uns = {
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

adata.obsm = {}
adata.obsp = {}
adata.varm = {}

adata.obs_names = adata.obs_names.str.replace("(-).*", "", regex=True) + "-e14.5-v2019"

adata

# %%
adata.write(SC_PROCESSED_DIR / "gex_e14.5.h5ad")

# %% [markdown]
# ### E15.5

# %%
adata = adata_rna[adata_rna.obs["day"].isin(["15.5"]), :]
adata.obs = adata.obs.loc[:, ["day", "celltype"]].replace(
    {
        "15.5": "E15.5",
        "Ngn3 High early": "Ngn3 high EP",
        "Ngn3 High late": "Ngn3 high EP",
        "Fev+ Pyy": "Eps/Delta progenitors",
        "Fev+ Epsilon": "Eps/Delta progenitors",
    }
)
adata.obs["celltype_fine"] = adata.obs["celltype"].copy()
adata.obs["celltype"].replace(
    {
        "Fev+ Alpha": "Pre-endocrine",
        "Fev+ Beta": "Pre-endocrine",
        "Fev+ Delta": "Pre-endocrine",
        "Eps/Delta progenitors": "Pre-endocrine",
    },
    inplace=True,
)
adata.obs["protocol"] = "scRNA-seq"
adata.obs = adata.obs.astype({"celltype": "category", "celltype_fine": "category"})
adata = adata[adata.obs["celltype"].isin(CELLTYPES_TO_KEEP), :].copy()

adata.var = adata.var.loc[:, []]

adata.uns = {
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

adata.obsm = {}
adata.obsp = {}
adata.varm = {}

adata.obs_names = adata.obs_names.str.replace("(-).*", "", regex=True) + "-e15.5-v2019"

adata

# %%
adata.write(SC_PROCESSED_DIR / "gex_e15.5.h5ad")
