import logging
import warnings
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_ind

from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._module import VELOVAE

logger = logging.getLogger(__name__)


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv


class VELOVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """Velocity Variational Inference.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~velovi.VELOVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    nu_init_data
        Initialize nuclear export rate using the data-driven technique.
    gamma_init_data
        Initialize gamma using the data-driven technique.
    linear_decoder
        Use a linear decoder from latent space to time.
    **model_kwargs
        Keyword args for :class:`~velovi.VELOVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        nu_init_data: bool = False,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_latent = n_latent

        spliced_nuc = self.adata_manager.get_from_registry(REGISTRY_KEYS.S_NUC_KEY)
        spliced_cyt = self.adata_manager.get_from_registry(REGISTRY_KEYS.S_CYT_KEY)
        unspliced_nuc = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_NUC_KEY)

        us_nuc_upper, ms_nuc_upper, ms_cyt_upper = self.percentile_median(
            adata, unspliced_nuc, spliced_nuc, spliced_cyt
        )

        # alpha = transcription rate
        alpha_unconstr = _softplus_inverse(us_nuc_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        # nu = nuclear export rate
        if nu_init_data:
            nu_unconstr = np.clip(
                _softplus_inverse(us_nuc_upper / ms_nuc_upper), None, 10
            )
        else:
            nu_unconstr = None

        # gamma = degradation rate
        if gamma_init_data:
            gamma_unconstr = np.clip(
                _softplus_inverse(us_nuc_upper / ms_cyt_upper), None, 10
            )
        else:
            gamma_unconstr = None

        self.module = VELOVAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            nu_unconstr_init=nu_unconstr,
            switch_spliced_nuc=ms_nuc_upper,
            switch_spliced_cyt=ms_cyt_upper,
            switch_unspliced_nuc=us_nuc_upper,
            linear_decoder=linear_decoder,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VELOVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

    def percentile_median(
        self,
        adata: AnnData,
        unspliced_nuc: np.ndarray,
        spliced_nuc: np.ndarray,
        spliced_cyt: np.ndarray,
    ):
        """Calculates the median of all abundances of cells which are above 99th percentile of unspliced abundance.

        Parameters
        ----------
        unspliced_nuc
            unspliced abundances in nucleus
        spliced_nuc
            spliced abundances in nucleus
        spliced_cyt
            spliced abundances in cytoplasm
        """
        argsort_unspliced_nuc = np.argsort(unspliced_nuc, axis=0)
        start_id_99_percentile = int(adata.n_obs * 0.99)
        us_upper_ids = argsort_unspliced_nuc[start_id_99_percentile:, :]

        us_nuc_upper = []
        ms_nuc_upper = []
        ms_cyt_upper = []
        # TODO: can be parallelized
        for obs_id in us_upper_ids:
            us_nuc_upper += [
                unspliced_nuc[obs_id, np.arange(adata.n_vars)][np.newaxis, :]
            ]
            ms_nuc_upper += [
                spliced_nuc[obs_id, np.arange(adata.n_vars)][np.newaxis, :]
            ]
            ms_cyt_upper += [
                spliced_cyt[obs_id, np.arange(adata.n_vars)][np.newaxis, :]
            ]
        us_nuc_upper = np.median(np.concatenate(us_nuc_upper, axis=0), axis=0)
        ms_nuc_upper = np.median(np.concatenate(ms_nuc_upper, axis=0), axis=0)
        ms_cyt_upper = np.median(np.concatenate(ms_cyt_upper, axis=0), axis=0)

        return us_nuc_upper, ms_nuc_upper, ms_cyt_upper

    def train(
        self,
        max_epochs: Optional[int] = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        gradient_clip_val: float = 10,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        lr
            Learning rate for optimization
        weight_decay
            Weight decay for optimization
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        gradient_clip_val
            Val for gradient clipping
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        user_plan_kwargs = plan_kwargs.copy() if isinstance(plan_kwargs, dict) else {}
        plan_kwargs = {"lr": lr, "weight_decay": weight_decay, "optimizer": "AdamW"}
        plan_kwargs.update(user_plan_kwargs)

        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = {"gradient_clip_val": gradient_clip_val}
        trainer_kwargs.update(user_train_kwargs)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.inference_mode()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                # rep_steady_prob = pi[..., 3]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        # + rep_steady_prob * self.module.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.inference_mode()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal[
            "spliced_cyt", "spliced_nuc", "unspliced_nuc"
        ] = "spliced_cyt",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                alpha = inference_outputs["alpha"]
                beta = inference_outputs["beta"]
                nu = inference_outputs["nu"]
                gamma = inference_outputs["gamma"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                (
                    u_nuc_0,
                    s_nuc_0,
                    s_cyt_0,
                ) = self.module._get_induction_unspliced_spliced(
                    alpha, beta, nu, gamma, switch_time
                )
                rep_time = (self.module.t_max - switch_time) * tau

                (
                    mean_u_nuc_rep,
                    mean_s_nuc_rep,
                    mean_s_cyt_rep,
                ) = self.module._get_repression_unspliced_spliced(
                    u_nuc_0, s_nuc_0, s_cyt_0, beta, nu, gamma, rep_time
                )
                if velo_mode == "spliced_cyt":
                    velo_rep = nu * mean_s_nuc_rep - gamma * mean_s_cyt_rep
                elif velo_mode == "spliced_nuc":
                    velo_rep = beta * mean_u_nuc_rep - nu * mean_s_nuc_rep
                else:
                    velo_rep = alpha - beta * mean_u_nuc_rep

                ind_time = switch_time * rho
                (
                    mean_u_nuc_ind,
                    mean_s_nuc_ind,
                    mean_s_cyt_ind,
                ) = self.module._get_induction_unspliced_spliced(
                    alpha, beta, nu, gamma, ind_time
                )
                if velo_mode == "spliced_cyt":
                    velo_ind = nu * mean_s_nuc_ind - gamma * mean_s_cyt_ind
                elif velo_mode == "spliced_nuc":
                    velo_ind = beta * mean_u_nuc_ind - nu * mean_s_nuc_ind
                else:
                    velo_ind = alpha - beta * mean_u_nuc_ind

                velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced_cyt = self.adata_manager.get_from_registry(REGISTRY_KEYS.S_CYT_KEY)

        if clip:
            velos = np.clip(velos, -spliced_cyt[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return velos

    @torch.inference_mode()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        restrict_to_latent_dim: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s_cyt = []
        fits_s_nuc = []
        fits_u_nuc = []
        for tensors in scdl:
            minibatch_samples_s_cyt = []
            minibatch_samples_s_nuc = []
            minibatch_samples_u_nuc = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                    generative_kwargs={"latent_dim": restrict_to_latent_dim},
                )

                gamma = inference_outputs["gamma"]
                nu = inference_outputs["nu"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s_cyt,
                    mixture_dist_s_nuc,
                    mixture_dist_u_nuc,
                    _,
                ) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    nu,
                    beta,
                    alpha,
                )
                fit_s_cyt = mixture_dist_s_cyt.mean
                fit_s_nuc = mixture_dist_s_nuc.mean
                fit_u_nuc = mixture_dist_u_nuc.mean

                fit_s_cyt = fit_s_cyt[..., gene_mask]
                fit_s_cyt = fit_s_cyt.cpu().numpy()
                fit_s_nuc = fit_s_nuc[..., gene_mask]
                fit_s_nuc = fit_s_nuc.cpu().numpy()
                fit_u_nuc = fit_u_nuc[..., gene_mask]
                fit_u_nuc = fit_u_nuc.cpu().numpy()

                minibatch_samples_s_cyt.append(fit_s_cyt)
                minibatch_samples_s_nuc.append(fit_s_nuc)
                minibatch_samples_u_nuc.append(fit_u_nuc)

            # samples by cells by genes
            fits_s_cyt.append(np.stack(minibatch_samples_s_cyt, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s_cyt[-1] = np.mean(fits_s_cyt[-1], axis=0)
            # samples by cells by genes
            fits_s_nuc.append(np.stack(minibatch_samples_s_nuc, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s_nuc[-1] = np.mean(fits_s_nuc[-1], axis=0)
            # samples by cells by genes
            fits_u_nuc.append(np.stack(minibatch_samples_u_nuc, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u_nuc[-1] = np.mean(fits_u_nuc[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s_cyt = np.concatenate(fits_s_cyt, axis=-2)
            fits_s_nuc = np.concatenate(fits_s_nuc, axis=-2)
            fits_u_nuc = np.concatenate(fits_u_nuc, axis=-2)
        else:
            fits_s_cyt = np.concatenate(fits_s_cyt, axis=0)
            fits_s_nuc = np.concatenate(fits_s_nuc, axis=0)
            fits_u_nuc = np.concatenate(fits_u_nuc, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s_cyt = pd.DataFrame(
                fits_s_cyt,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_s_nuc = pd.DataFrame(
                fits_s_nuc,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u_nuc = pd.DataFrame(
                fits_u_nuc,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s_cyt, df_s_nuc, df_u_nuc
        else:
            return fits_s_cyt, fits_s_nuc, fits_u_nuc

    @torch.inference_mode()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the likelihood per gene. Higher is better.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                spliced_cyt = tensors[REGISTRY_KEYS.S_CYT_KEY]
                spliced_nuc = tensors[REGISTRY_KEYS.S_NUC_KEY]
                unspliced_nuc = tensors[REGISTRY_KEYS.U_NUC_KEY]

                gamma = inference_outputs["gamma"]
                nu = inference_outputs["nu"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s_cyt,
                    mixture_dist_s_nuc,
                    mixture_dist_u_nuc,
                    _,
                ) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    nu,
                    beta,
                    alpha,
                )
                device = gamma.device
                reconst_loss_s_cyt = -mixture_dist_s_cyt.log_prob(
                    spliced_cyt.to(device)
                )
                reconst_loss_s_nuc = -mixture_dist_s_nuc.log_prob(
                    spliced_nuc.to(device)
                )
                reconst_loss_u_nuc = -mixture_dist_u_nuc.log_prob(
                    unspliced_nuc.to(device)
                )
                output = -(reconst_loss_s_cyt + reconst_loss_s_nuc + reconst_loss_u_nuc)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls

    @torch.inference_mode()
    def get_rates(self):
        (
            gamma,
            nu,
            beta,
            alpha,
        ) = self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "nu": nu.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer_cyt: str,
        spliced_layer_nuc: str,
        unspliced_layer_nuc: str,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression.

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.S_CYT_KEY, spliced_layer_cyt, is_count_data=False),
            LayerField(REGISTRY_KEYS.S_NUC_KEY, spliced_layer_nuc, is_count_data=False),
            LayerField(
                REGISTRY_KEYS.U_NUC_KEY, unspliced_layer_nuc, is_count_data=False
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def get_directional_uncertainty(self):
        raise NotImplementedError

    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None
    ) -> Tuple[pd.DataFrame, AnnData]:
        """Compute permutation scores.

        Parameters
        ----------
        labels_key
            Key in adata.obs encoding cell types
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.

        Returns
        -------
        Tuple of DataFrame and AnnData. DataFrame is genes by cell types with score per cell type.
        AnnData is the permutated version of the original AnnData.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        # shuffle spliced then unspliced
        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.U_NUC_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.S_NUC_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.S_CYT_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        ms_cyt_ = adata_manager.get_from_registry(REGISTRY_KEYS.S_CYT_KEY)
        ms_nuc_ = adata_manager.get_from_registry(REGISTRY_KEYS.S_NUC_KEY)
        mu_nuc_ = adata_manager.get_from_registry(REGISTRY_KEYS.U_NUC_KEY)

        ms_cyt_p = bdata_manager.get_from_registry(REGISTRY_KEYS.S_CYT_KEY)
        ms_nuc_p = bdata_manager.get_from_registry(REGISTRY_KEYS.S_NUC_KEY)
        mu_nuc_p = bdata_manager.get_from_registry(REGISTRY_KEYS.U_NUC_KEY)

        spliced_cyt_, spliced_nuc_, unspliced_nuc_ = self.get_expression_fit(
            adata, n_samples=10
        )
        root_squared_error = np.abs(spliced_cyt_ - ms_cyt_)
        root_squared_error += np.abs(spliced_nuc_ - ms_nuc_)
        root_squared_error += np.abs(unspliced_nuc_ - mu_nuc_)

        spliced_cyt_p, spliced_nuc_p, unspliced_nuc_p = self.get_expression_fit(
            bdata, n_samples=10
        )
        root_squared_error_p = np.abs(spliced_cyt_p - ms_cyt_p)
        root_squared_error_p += np.abs(spliced_nuc_p - ms_nuc_p)
        root_squared_error_p += np.abs(unspliced_nuc_p - mu_nuc_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            for g in adata.var_names.tolist():
                x = root_squared_error_p[g][adata.obs[labels_key] == ct]
                y = root_squared_error[g][adata.obs[labels_key] == ct]
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata

    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        # get registry info to later set data back in bdata
        # in a way that doesn't require actual knowledge of location
        data_layer = bdata_manager.get_from_registry(registry_key)
        data_registry = bdata_manager.data_registry[registry_key]
        attr_name = data_registry.attr_name
        attr_key = data_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            data_layer_ct = data_layer[mask].copy()
            data_layer_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=data_layer_ct
            )
            data_layer[mask] = data_layer_ct
        # e.g., if using adata.X
        if attr_key is None:
            setattr(bdata, attr_name, data_layer)
        # e.g., if using a layer
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = data_layer
            setattr(bdata, attr_name, attribute)

        return bdata
