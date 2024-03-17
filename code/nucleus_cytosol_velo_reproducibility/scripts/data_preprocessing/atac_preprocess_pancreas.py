# %% [markdown]
# # Data Preprocessing
#
# This notebook reads in Pancreas multiome data, removes redundant information and extracts the ATAC modality information which is needed for scglue to build a common latent space of scATAC and scRNA measurements.
#
# **Requires:**
# - `/vol/storage/data/pancreas_multiome/raw/multiome/pancreas_multiome_2022_processed.h5ad`
#
# **Output:**
# - `/vol/storage/data/pancreas_multiome/processed/atac_e14.5.h5ad`
# - `/vol/storage/data/pancreas_multiome/processed/atac_e15.5.h5ad`

# %% [markdown]
# ## Library Imports

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
    "Eps/Delta progenitors",
    "Epsilon",
    "Fev+",
    "Fev+ Alpha",
    "Fev+ Beta",
    "Fev+ Delta",
    "Ngn3 high",
    "Ngn3 low",
]

SN_RAW_DIR = PROJECT_DIR / "pancreas_multiome" / "raw" / "multiome"
SN_PROCESSED_DIR = PROJECT_DIR / "pancreas_multiome" / "processed"

# If Processed folder doesnt exist, create it first
SN_PROCESSED_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
# Read in multiome data
adata_rna = sc.read(SN_RAW_DIR / "pancreas_multiome_2022_processed.h5ad")
adata_rna

# %% [markdown]
# ## Data processing and saving

# %% [markdown]
# ### E14.5

# %%
adata = adata_rna[adata_rna.obs["sample"].isin(["E14.5"]), :].copy()
atac_vars = adata.var_names[adata.var["modality"] == "ATAC"]
adata = adata[:, list(atac_vars)]

adata = adata[adata.obs["celltype"].isin(CELLTYPES_TO_KEEP), :].copy()
adata.obs = (
    adata.obs.loc[:, ["sample", "celltype"]]
    .rename({"sample": "day"}, axis=1)
    .replace(
        {
            "Ngn3 high": "Ngn3 high EP",
            "Ngn3 low": "Ngn3 low EP",
        }
    )
)
adata.obs["celltype_fine"] = adata.obs["celltype"].copy()
adata.obs["celltype"].replace(
    {
        "Fev+": "Pre-endocrine",
        "Fev+ Alpha": "Pre-endocrine",
        "Fev+ Beta": "Pre-endocrine",
        "Fev+ Delta": "Pre-endocrine",
        "Eps/Delta progenitors": "Pre-endocrine",
    },
    inplace=True,
)

adata.obs = adata.obs.astype({"celltype": "category", "celltype_fine": "category"})
adata.obsm = {}
adata.varm = {}
adata.layers = {}
adata.obsp = {}
adata.obs_names = adata.obs_names.str.replace("(-).*", "", regex=True) + "-e14.5-v2022"
adata.layers["count"] = adata.X

# Add color definitions for celltypes at very end
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
        "Fev+": "#fbafe4",
        "Ngn3 high EP": "#fdbf6f",
        "Ngn3 low EP": "#f4a460",
        "Fev+ Alpha": "#d55e00",
        "Fev+ Beta": "#cc78bc",
        "Fev+ Delta": "#ca9161",
    },
}

adata

# %%
adata.write(SN_PROCESSED_DIR / "atac_e14.5.h5ad")

# %% [markdown]
# ### E15.5

# %%
adata = adata_rna[adata_rna.obs["sample"].isin(["E15.5"]), :].copy()
atac_vars = adata.var_names[adata.var["modality"] == "ATAC"]
adata = adata[:, list(atac_vars)]

adata = adata[adata.obs["celltype"].isin(CELLTYPES_TO_KEEP), :].copy()
adata.obs = (
    adata.obs.loc[:, ["sample", "celltype"]]
    .rename({"sample": "day"}, axis=1)
    .replace(
        {
            "Ngn3 high": "Ngn3 high EP",
            "Ngn3 low": "Ngn3 low EP",
        }
    )
)
adata.obs["celltype_fine"] = adata.obs["celltype"].copy()
adata.obs["celltype"].replace(
    {
        "Fev+": "Pre-endocrine",
        "Fev+ Alpha": "Pre-endocrine",
        "Fev+ Beta": "Pre-endocrine",
        "Fev+ Delta": "Pre-endocrine",
        "Eps/Delta progenitors": "Pre-endocrine",
    },
    inplace=True,
)

adata.obs = adata.obs.astype({"celltype": "category", "celltype_fine": "category"})
adata.obsm = {}
adata.varm = {}
adata.layers = {}
adata.obsp = {}
adata.obs_names = adata.obs_names.str.replace("(-).*", "", regex=True) + "-e15.5-v2022"
adata.layers["count"] = adata.X

# Add color definitions for celltypes at very end
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
        "Fev+": "#fbafe4",
        "Ngn3 high EP": "#fdbf6f",
        "Ngn3 low EP": "#f4a460",
        "Fev+ Alpha": "#d55e00",
        "Fev+ Beta": "#cc78bc",
        "Fev+ Delta": "#ca9161",
    },
}

adata

# %%
adata.write(SN_PROCESSED_DIR / "atac_e15.5.h5ad")
