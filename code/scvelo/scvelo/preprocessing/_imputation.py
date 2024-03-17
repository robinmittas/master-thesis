from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import muon as mu

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from pandas.api.types import is_categorical_dtype

import scanpy as sc
from anndata import AnnData

from scvelo import logging as logg
from .utils import filter_genes, filter_genes_dispersion, log1p, normalize_per_cell

ArrayLike = Union[sp.spmatrix, np.ndarray]


def get_shared_obs_and_vars(adata_atac, adata_rna):
    """Returns shared ob_names and var_names between 2 adata objects.

    Parameters
    ----------
    adata_atac
        Annotated data object 1
    adata_atac
        Annotated data object 2
    """
    shared_obs = pd.Index(np.intersect1d(adata_rna.obs_names, adata_atac.obs_names))
    shared_vars = pd.Index(np.intersect1d(adata_rna.var_names, adata_atac.var_names))

    return shared_obs, shared_vars


def get_wnn(mdata, n_svd_comps=50, n_pca_comps=50, n_neighbors=50):
    """Returns cell-cell weighted distance matrix.

    Parameters
    ----------
    mdata
        mudata object consisting of multi-omics data
    n_svd_comps
        Number of SVD components for ATAC measurements (sparse)
    n_pca_comps
        Number of PCA components for RNA-seq measurements
    n_neighbors
        Number of neigbor cells to consider
    """
    # Normalize data and find variable features
    filter_genes(mdata["rna"])  # not used in Seurat but close
    normalize_per_cell(mdata["rna"])
    filter_genes_dispersion(
        mdata["rna"]
    )  # not exactly same filtering as Seurat defaults
    mdata["rna"].X = mdata["rna"].X * 10000
    log1p(mdata["rna"])

    # Scale data
    mdata["rna"].X = mdata["rna"].X - mdata["rna"].X.mean(axis=0)  # centre about zero
    # Run PCA
    sc.tl.pca(mdata["rna"], n_comps=n_pca_comps)
    # Run TFIDF
    mu.atac.pp.tfidf(mdata["atac"], log_idf=False, log_tf=False, log_tfidf=True)

    # Find top features runs with q0 => does nothing since 100th percentile
    # Run SVD
    svd = sklearn.decomposition.TruncatedSVD(
        n_components=n_svd_comps
    )  # doesn't use irlba like Seurat
    out = svd.fit(mdata["atac"].X.T)
    mdata["atac"].obsm["X_svd"] = out.components_.T

    sc.pp.neighbors(mdata["rna"], use_rep="X_pca", n_neighbors=n_neighbors)
    sc.pp.neighbors(mdata["atac"], use_rep="X_svd", n_neighbors=n_neighbors)

    # Calculate weighted nearest neighbors
    mu.pp.neighbors(mdata, key_added="wnn", n_neighbors=n_neighbors)

    nn_idx = mdata.obsp["wnn_distances"].tolil().rows.tolist()
    nn_idx = np.array(nn_idx)
    nn_dist = np.zeros(nn_idx.shape)
    for i in range(nn_idx.flatten().max() + 1):
        nn_dist[i, :] = mdata.obsp["wnn_distances"][i, nn_idx[i]].toarray()[0]

    return nn_idx, nn_dist


def _smooth(
    data: ArrayLike,
    graph: sp.csr_matrix,
    *,
    add_diagonal: bool = False,
    modality: str,
) -> ArrayLike:
    """Function smooths data by considereing neighbor-cells.

    Checks if graph is valid, calculates the numer of neighbor cells, gives an error if data has no `modality` neighbor-cell and returns a weighted average of neighbor cell's abundance given the graphs weights.

    Parameters
    ----------
    data
        ArrayLike data containing e.g. count spliced/ unspliced abundance
    graph
        Cell-cell neighbor graph
    add_diagonal
        Sets diagonal to 1 and checks if graph is valid
    modality
        "scRNA"/ "snRNA"
    """
    if add_diagonal:
        assert graph.shape[0] == graph.shape[1], "Graph must be a square matrix."
        graph = graph.copy()
        graph.setdiag(1.0)
    graph = graph > 0.0
    num_total_neighbors = graph.sum(axis=1).A1
    has_neighbors = num_total_neighbors != 0
    if not np.all(has_neighbors):
        msg = (
            f"`{np.sum(~has_neighbors)}/{len(has_neighbors)}` cell(s) "
            f"have no `{modality}` neighbors."
        )
        raise ValueError(msg)

    inv_num_neighbors = sp.diags(1.0 / num_total_neighbors)  # (k, k)
    return (inv_num_neighbors @ graph) @ data  # (k, m) @ (m, f) - > (k, f)


def _estimate_missing(
    features: ArrayLike,
    *,
    graph: sp.csr_matrix,
    g_mask: np.ndarray,
    modality: str,
) -> Tuple[ArrayLike, ArrayLike]:
    """Function estimates missing abundance.

    Parameters
    ----------
    features
        ArrayLike data containing e.g. count spliced/ unspliced abundance
    graph
        Cell-cell neighbor graph
    g_mask
        Mask with True/False values indicating  whether cell is a `modality` measure
    modality
        "scRNA"/ "snRNA"
    """
    # invariant: n = k + m
    g = features[g_mask]  # (n, f) -> (m, f)

    g_missing = ~g_mask  # (n,)
    sub_graph = graph[g_missing, :][:, g_mask]  # (k, m)
    g_hat = _smooth(features[g_mask], sub_graph, modality=modality)  # (k, f)

    assert g.shape[0] + g_hat.shape[0] == features.shape[0]
    return g, g_hat


def _recombine(
    *, g: ArrayLike, g_hat: ArrayLike, g_mask: ArrayLike, g_hat_mask: ArrayLike
) -> ArrayLike:
    """Function combines true values where known and estimated values where the data is missing.

    Parameters
    ----------
    g
        ArrayLike data containing known/ measured count spliced/ unspliced abundance
    g_hat
        ArrayLike data containing unknown/ estimated count spliced/ unspliced abundance
    g_mask
        Mask with True/False values indicating  where to keep g
    g_hat_mask
        Mask with True/False values indicating  where to keep g
    """
    assert g.shape[1] == g_hat.shape[1]
    shape = g.shape[0] + g_hat.shape[0], g.shape[1]

    g_res = sp.csr_matrix(shape) if sp.issparse(g) else np.zeros(shape)
    g_res[g_mask, :] = g
    g_res[g_hat_mask, :] = g_hat

    return g_res


def _subset_min_neighbors(
    adata: AnnData, *, n: int, key: str, missing_mask: np.ndarray
) -> AnnData:
    """Function subsets adata object s.t. each cell has at least `n` neighbors of different modality.

    Parameters
    ----------
    adata
        AnnData object
    n
        Minimum number of neighbors
    key
        Key/ mode where graph is stored (e.g. 'connectivities', 'distances')
    missing_mask
        Mask with True/False values indicating  where values are missing
    """
    assert np.issubdtype(missing_mask, bool)

    G = adata.obsp[key] > 0.0
    existing_mask = ~missing_mask
    num_neighs = G[missing_mask, :][:, existing_mask].sum(1).A1  # e.g., sc -> sn

    keep_mask = np.ones((adata.n_obs,), dtype=bool)
    keep_mask[missing_mask] = (
        num_neighs >= n
    )  # e.g., keep only sc that have enough sn neighbors

    return adata[keep_mask]


def _fixed_point(
    adata: AnnData, *, n: int, key: str, dataset_key: str, sc_rna_name: str
) -> AnnData:
    """Function subsets adata object, e.g. cells, s.t. each cell has at least `n` neighbors of different modality.

    Parameters
    ----------
    adata
        AnnData object
    n
        Minimum number of neighbors
    key
        Key/ mode where graph is stored (e.g. 'connectivities', 'distances')
    dataset_key
        Key in :attr:`anndata.AnnData.obs` where the modality indicator
    sc_rna_name
        Name of the first modality, e.g., scRNA-seq,
        in :attr:`anndata.AnnData.obs` ``['{dataset_key}']``.
    """
    orig_n_obs = adata.n_obs
    while adata.n_obs:
        adata_prev = adata
        adata = _subset_min_neighbors(
            adata_prev,
            n=n,
            key=key,
            missing_mask=adata_prev.obs[dataset_key] == sc_rna_name,
        )
        adata = _subset_min_neighbors(
            adata, n=n, key=key, missing_mask=adata.obs[dataset_key] != sc_rna_name
        )

        if adata.n_obs == adata_prev.n_obs:
            break

    if not adata.n_obs:
        raise ValueError(f"Unable to find fixpoint, decrease `n={n}`.")

    logg.warn(
        f"After running fixed-point iteration, "
        f"`{adata.n_obs}/{orig_n_obs}` observations remain"
    )

    return adata


def _estimate_lambda(
    *,
    sc_rna: ArrayLike,
    sn_rna: ArrayLike,
    update_zero_sn: bool = False,
    update_zero_sc: bool = False,
) -> ArrayLike:
    r"""Estimate the ratio ``scRNA-seq / snRNA-seq``.

    Parameters
    ----------
    sc_rna
        scRNA abundance
    sn_rna
        snRNA abundance
    update_zero_sn
        Boolean indicator to set ratio to 1, where snRNA is 0
    update_zero_sc
        Boolean indicator to set ratio to 1, where scRNA is 0
    """

    def densify(x: ArrayLike) -> np.ndarray:
        return x.A if sp.issparse(x) else x

    sparsify = sp.issparse(sn_rna) and sp.issparse(sc_rna)
    sn_rna = densify(sn_rna)
    sc_rna = densify(sc_rna)

    sn_zeros = sn_rna == 0.0
    sc_zeros = sc_rna == 0.0

    # TODO(michalk8): sparse implementation
    lam = np.zeros_like(sc_rna)
    if update_zero_sn:
        lam[~sc_zeros & sn_zeros] = 1.0
    if update_zero_sc:
        lam[sc_zeros & ~sn_zeros] = 1.0
    mask = ~sc_zeros & ~sn_zeros
    lam[mask] = sc_rna[mask] / sn_rna[mask]

    return sp.csr_matrix(lam) if sparsify else lam


def _require_non_negative(x: ArrayLike) -> ArrayLike:
    """Clips array where negative to 0.

    Parameters
    ----------
    x
        ArrayLike object containing abundances.
    """
    if sp.isspmatrix(x):
        logg.warn(
            f"Removing `{100 * np.sum(x.data < 0) / x.data.shape[0]:.4f}%` "
            f"elements below 0"
        )
        x = x.copy()
        x.data[:] = np.clip(x.data, 0.0, None)
    else:
        logg.warn(
            f"Removing `{100 * np.sum(x < 0) / np.prod(x.shape):.4f}%` "
            f"elements below 0"
        )
        x = np.clip(x, 0.0, None)
    return x


def estimate_abundance(
    adata: AnnData,
    mode: Literal["connectivities", "distances"] = "connectivities",
    layers: Union[Optional[str], Sequence[Optional[str]]] = ("spliced", "unspliced"),
    dataset_key: str = "modality",
    sc_rna_name: str = "scRNA-seq",
    min_estimation_samples: int = 1,
    smooth_obs: bool = True,
    clip_cyto: bool = True,
    lambda_correction: bool = True,
    estimate_cyto: bool = True,
    filter_zero_genes: bool = True,
) -> Optional[AnnData]:
    """Estimate scRNA-seq from snRNA-seq data and vice-versa.

    Also estimate the amount of RNA in the cytoplasm.

    Parameters
    ----------
    adata
        Annotated data object.
    mode
        Key in :attr:`anndata.AnnData.obsp` where the cell-cell graph is stored.
    layers
        Keys in :attr:`anndata.AnnData.layers` or `None` for :attr:`anndata.AnnData.X`
        for which to estimate the abundance.
    dataset_key
        Key in :attr:`anndata.AnnData.obs` where the modality indicator
        (e.g., snRNA-seq vs. scRNA-seq) is stored. Currently, only 2 modalities are supported.
    sc_rna_name
        Name of the first modality, e.g., scRNA-seq,
        in :attr:`anndata.AnnData.obs` ``['{dataset_key}']``.
    min_estimation_samples
        Minimum number of samples to estimate the missing modality.
    smooth_obs
        Whether to smooth the observed modalities using
        :attr:`anndata.AnnData.obsp` ``['{mode}']``.
    clip_cyto
        Whether to clip negative values in the cytoplasm RNA to 0.
    lambda_correction
        Whether to estimate the ratio ``lambda = scRNA-seq-uns / snRNA-seq-uns`` and update
        ``scRNA-seq-spl = lambda * scRNA-seq-spl`` when estimating the RNA in cytoplasm.
    estimate_cyto
        Whether to estimate spliced RNA in the cytoplasm.
        Requires `'spliced'` and `'unspliced'` to be present in ``layers``.
    filter_zero_genes
        Whether to drop genes where lambda_correction is all 0 and thus has no (un)spliced counts in nucleus.
        Requires `'lambda_correction'` to be set to True.

    Returns
    -------
    `None` if ``inplace = True``, otherwise the modified ``adata``.
    The following keys will be added, for each key in ``layers``:

        - :attr:`anndata.AnnData.layers` ``['{layer}_cell']`` - the observed and estimated whole-cell RNA.
        - :attr:`anndata.AnnData.layers` ``['{layer}_nucleus']`` - the observed and estimated nucleus RNA.
        - :attr:`anndata.AnnData.layers` ``['{layer}_cytoplasm']`` - the estimated cytoplasmic RNA.
    """
    if layers is None or isinstance(layers, str):
        layers = (layers,)
    layers = set(layers)

    inplace = False
    if min_estimation_samples:
        adata = _fixed_point(
            adata,
            n=min_estimation_samples,
            key=mode,
            dataset_key=dataset_key,
            sc_rna_name=sc_rna_name,
        )
        inplace = False
    if not inplace:
        adata = adata.copy()

    dataset = adata.obs[dataset_key]
    if not is_categorical_dtype(dataset):
        dataset = dataset.astype("category")
    assert len(dataset.cat.categories) == 2, "Only 2 categories are supported."

    cell_mask = (dataset == sc_rna_name).values  # (n,); m elements
    nuc_mask = ~cell_mask  # (n,); k elements

    assert np.sum(cell_mask), "No data for whole-cell RNA."
    assert np.sum(nuc_mask), "No data for nucleic RNA."

    graph = adata.obsp[mode]  # (n, n)
    if not sp.isspmatrix_csr(graph):
        graph = graph.tocsr()

    data: Dict[str, ArrayLike] = {}
    for layer in layers:
        features = adata.X if layer is None else adata.layers[layer]
        # (m, f), (k, f)
        g_cell, g_cell_hat = _estimate_missing(
            features, graph=graph, g_mask=cell_mask, modality="scRNA"
        )
        # (k, f), (m, f)
        g_nuc, g_nuc_hat = _estimate_missing(
            features, graph=graph, g_mask=nuc_mask, modality="snRNA"
        )
        if smooth_obs:
            g_cell = _smooth(
                g_cell,
                graph[cell_mask, :][:, cell_mask],
                add_diagonal=True,
                modality="scRNA",
            )  # (m, f)
            g_nuc = _smooth(
                g_nuc,
                graph[nuc_mask, :][:, nuc_mask],
                add_diagonal=True,
                modality="snRNA",
            )  # (k, f)

        cell = _recombine(
            g=g_cell, g_hat=g_cell_hat, g_mask=cell_mask, g_hat_mask=nuc_mask
        )  # (n, f)
        nuc = _recombine(
            g=g_nuc, g_hat=g_nuc_hat, g_mask=nuc_mask, g_hat_mask=cell_mask
        )  # (n, f)

        layer = "X" if layer is None else layer
        data[f"{layer}_cell"] = cell
        data[f"{layer}_nucleus"] = nuc

    if estimate_cyto:
        if lambda_correction:
            sc_rna_u = data["unspliced_cell"]
            sn_rna_u = data["unspliced_nucleus"]
            data["lambda"] = lam = _estimate_lambda(sc_rna=sc_rna_u, sn_rna=sn_rna_u)
            data["unspliced_nucleus_original"] = data["unspliced_nucleus"]
            data["spliced_nucleus_original"] = data["spliced_nucleus"]
            if sp.isspmatrix(lam):
                data["unspliced_nucleus"] = lam.multiply(data["unspliced_nucleus"])
                data["spliced_nucleus"] = lam.multiply(data["spliced_nucleus"])
            else:
                data["unspliced_nucleus"] *= lam
                data["spliced_nucleus"] *= lam

        cyt = data["spliced_cell"] - data["spliced_nucleus"]
        data["spliced_cytoplasm"] = _require_non_negative(cyt) if clip_cyto else cyt

    adata.layers = {**adata.layers, **data}

    if lambda_correction and filter_zero_genes:
        # check for which genes 'unspliced_cell/unspliced_nucleus' is 0 for all cells
        lambda_sum = np.array(adata.layers["lambda"].sum(axis=0)).squeeze()
        adata = adata[:, lambda_sum != 0]

    if not inplace:
        return adata
