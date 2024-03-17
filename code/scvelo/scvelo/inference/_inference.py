import torch

from anndata import AnnData

from ._model import VELOVI


def fit_velovi(
    adata: AnnData,
    unspliced_layer_nuc: str = "unspliced_nucleus",
    spliced_layer_nuc: str = "spliced_nucleus",
    spliced_layer_cyt: str = "spliced_cytosol",
    max_epochs: int = 2000,
    n_samples: int = 25,
):
    """Function returns spliced RNA abundance in cytoplasm with the given solution to the ODE.

    Parameters
    ----------
    adata
        AnnData object
    unspliced_layer_nuc
        Name of unspliced layer in nucleus of adata object
    spliced_layer_nuc
        Name of spliced layer in nucleus of adata object
    spliced_layer_cyt
        Name of spliced layer in cytosol of adata object
    max_epochs
        Maximal Numbers of epochs to train velovi
    n_samples
        Number of posterior samples to use for estimation.
    """
    VELOVI.setup_anndata(
        adata,
        spliced_layer_nuc=spliced_layer_nuc,
        unspliced_layer_nuc=unspliced_layer_nuc,
        spliced_layer_cyt=spliced_layer_cyt,
    )

    vae = VELOVI(adata)
    vae.train(max_epochs=max_epochs)

    latent_time = vae.get_latent_time(n_samples=n_samples)

    adata.var["fit_alpha"] = vae.get_rates()["alpha"]
    adata.var["fit_beta"] = vae.get_rates()["beta"]
    adata.var["fit_nu"] = vae.get_rates()["nu"]
    adata.var["fit_gamma"] = vae.get_rates()["gamma"]
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    )
    adata.layers["fit_t"] = latent_time.values

    return vae, adata
