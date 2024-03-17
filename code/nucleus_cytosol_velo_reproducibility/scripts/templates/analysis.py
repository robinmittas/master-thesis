# %% [markdown]
# # Title
#
# Brief description.
#
# **Requires:** List of files required to run in order to execute this notebook.
#
# **Output:** List of outputted files

# %%
# %load_ext autoreload
# %autoreload 2

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
# Constants, plotting settings, etc.

# %% [markdown]
# ## Function definitions

# %%
# Helper functions

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(PROJECT_DIR / ".h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %%

# %% [markdown]
# ## Data analysis

# %%
