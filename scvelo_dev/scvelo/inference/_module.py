"""Main module."""
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet
from torch.distributions import kl_divergence as kl
from torch.distributions import MixtureSameFamily, Normal

from scvi.module.base import auto_move_data, BaseModuleClass, LossOutput
from scvi.nn import Encoder, FCLayers

from ._constants import REGISTRY_KEYS

torch.backends.cudnn.benchmark = True


class DecoderVELOVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # categorical pi
        # 4 states
        self.px_pi_decoder = nn.Linear(n_hidden, 4 * n_output)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        # The decoder returns values for the parameters of the ZINB distribution
        rho_first = self.rho_first_decoder(z_in)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp()
                + self.linear_scaling_tau_intercept
            )

        # cells by genes by 4
        pi_first = self.pi_first_decoder(z)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 4))
        )

        return px_pi, px_rho, px_tau


# VAE model
class VELOVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the veloVI model descibed in :cite:p:`GayosoWeiler2022`

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        nu_unconstr_init: Optional[np.ndarray] = None,
        switch_spliced_nuc: Optional[np.ndarray] = None,
        switch_spliced_cyt: Optional[np.ndarray] = None,
        switch_unspliced_nuc: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate

        if switch_spliced_nuc is not None:
            self.register_buffer(
                "switch_spliced_nuc", torch.from_numpy(switch_spliced_nuc)
            )
        else:
            self.switch_spliced_nuc = None

        if switch_spliced_cyt is not None:
            self.register_buffer(
                "switch_spliced_cyt", torch.from_numpy(switch_spliced_cyt)
            )
        else:
            self.switch_spliced_cyt = None

        if switch_unspliced_nuc is not None:
            self.register_buffer(
                "switch_unspliced_nuc", torch.from_numpy(switch_unspliced_nuc)
            )
        else:
            self.switch_unspliced_nuc = None

        n_genes = n_input * 3

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # nuclear export
        if nu_unconstr_init is None:
            self.nu_mean_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.nu_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(nu_unconstr_init)
            )

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
        )

    def _get_inference_input(self, tensors):
        spliced_nuc = tensors[REGISTRY_KEYS.S_NUC_KEY]
        spliced_cyt = tensors[REGISTRY_KEYS.S_CYT_KEY]
        unspliced_nuc = tensors[REGISTRY_KEYS.U_NUC_KEY]

        input_dict = {
            "spliced_nuc": spliced_nuc,
            "spliced_cyt": spliced_cyt,
            "unspliced_nuc": unspliced_nuc,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]

        gamma = inference_outputs["gamma"]
        nu = inference_outputs["nu"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "nu": nu,
            "beta": beta,
            "alpha": alpha,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced_nuc,
        spliced_cyt,
        unspliced_nuc,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_nuc_ = spliced_nuc
        spliced_cyt_ = spliced_cyt
        unspliced_nuc_ = unspliced_nuc
        if self.log_variational:
            spliced_nuc_ = torch.log(0.01 + spliced_nuc)
            spliced_cyt_ = torch.log(0.01 + spliced_cyt)
            unspliced_nuc_ = torch.log(0.01 + unspliced_nuc)

        encoder_input = torch.cat((spliced_cyt_, spliced_nuc_, unspliced_nuc_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, nu, beta, alpha = self._get_rates()

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "gamma": gamma,
            "nu": nu,
            "beta": beta,
            "alpha": alpha,
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        # nuclear export
        nu = torch.clamp(F.softplus(self.nu_mean_unconstr), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)

        return gamma, nu, beta, alpha

    @auto_move_data
    def generative(self, z, gamma, nu, beta, alpha, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(decoder_input, latent_dim=latent_dim)
        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        (
            mixture_dist_s_cyt,
            mixture_dist_s_nuc,
            mixture_dist_u_nuc,
            end_penalty,
        ) = self.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            nu,
            beta,
            alpha,
        )

        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "scale": scale,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u_nuc": mixture_dist_u_nuc,
            "mixture_dist_s_nuc": mixture_dist_s_nuc,
            "mixture_dist_s_cyt": mixture_dist_s_cyt,
            "end_penalty": end_penalty,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced_cyt = tensors[REGISTRY_KEYS.S_CYT_KEY]
        spliced_nuc = tensors[REGISTRY_KEYS.S_NUC_KEY]
        unspliced_nuc = tensors[REGISTRY_KEYS.U_NUC_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s_cyt = generative_outputs["mixture_dist_s_cyt"]
        mixture_dist_s_nuc = generative_outputs["mixture_dist_s_nuc"]
        mixture_dist_u_nuc = generative_outputs["mixture_dist_u_nuc"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s_cyt = -mixture_dist_s_cyt.log_prob(spliced_cyt)
        reconst_loss_s_nuc = -mixture_dist_s_nuc.log_prob(spliced_nuc)
        reconst_loss_u_nuc = -mixture_dist_u_nuc.log_prob(unspliced_nuc)

        reconst_loss = (
            reconst_loss_u_nuc.sum(dim=-1)
            + reconst_loss_s_nuc.sum(dim=-1)
            + reconst_loss_s_cyt.sum(dim=-1)
        )

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty

        loss_recoder = LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

        return loss_recoder

    @auto_move_data
    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        nu,
        beta,
        alpha,
    ) -> torch.Tensor:
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        (
            mean_u_nuc_ind,
            mean_s_nuc_ind,
            mean_s_cyt_ind,
        ) = self._get_induction_unspliced_spliced(alpha, beta, nu, gamma, t_s * px_rho)

        # induction steady state
        (
            mean_u_nuc_ind_steady,
            mean_s_nuc_ind_steady,
            mean_s_cyt_ind_steady,
        ) = self._get_induction_steady_states(alpha, beta, nu, gamma, n_cells)

        scale_u_nuc = scale[: self.n_input, :].expand(n_cells, self.n_input, 4).sqrt()
        scale_s_nuc = (
            scale[self.n_input : 2 * self.n_input, :]
            .expand(n_cells, self.n_input, 4)
            .sqrt()
        )
        scale_s_cyt = (
            scale[2 * self.n_input :, :].expand(n_cells, self.n_input, 4).sqrt()
        )

        # repression (initial condition: px_rho->1)
        u_nuc_0, s_nuc_0, s_cyt_0 = self._get_induction_unspliced_spliced(
            alpha, beta, nu, gamma, t_s
        )

        tau = px_tau
        (
            mean_u_nuc_rep,
            mean_s_nuc_rep,
            mean_s_cyt_rep,
        ) = self._get_repression_unspliced_spliced(
            u_nuc_0,
            s_nuc_0,
            s_cyt_0,
            beta,
            nu,
            gamma,
            (self.t_max - t_s) * tau,
        )
        # repression steady state
        mean_u_nuc_rep_steady = torch.zeros_like(mean_u_nuc_ind)
        mean_s_nuc_rep_steady = torch.zeros_like(mean_u_nuc_ind)
        mean_s_cyt_rep_steady = torch.zeros_like(mean_u_nuc_ind)

        # end penalty (L_switch)
        end_penalty = ((u_nuc_0 - self.switch_unspliced_nuc).pow(2)).sum() + (
            (s_nuc_0 - self.switch_spliced_nuc).pow(2)
            + ((s_cyt_0 - self.switch_spliced_cyt).pow(2))
        ).sum()

        # unspliced nucleus
        mean_u_nuc = torch.stack(
            (
                mean_u_nuc_ind,
                mean_u_nuc_ind_steady,
                mean_u_nuc_rep,
                mean_u_nuc_rep_steady,
            ),
            dim=2,
        )
        scale_u_nuc = torch.stack(
            (
                scale_u_nuc[..., 0],
                scale_u_nuc[..., 0],
                scale_u_nuc[..., 0],
                0.1 * scale_u_nuc[..., 0],
            ),
            dim=2,
        )
        dist_u_nuc = Normal(mean_u_nuc, scale_u_nuc)
        mixture_dist_u_nuc = MixtureSameFamily(comp_dist, dist_u_nuc)

        # spliced nucleus
        mean_s_nuc = torch.stack(
            (
                mean_s_nuc_ind,
                mean_s_nuc_ind_steady,
                mean_s_nuc_rep,
                mean_s_nuc_rep_steady,
            ),
            dim=2,
        )
        scale_s_nuc = torch.stack(
            (
                scale_s_nuc[..., 0],
                scale_s_nuc[..., 0],
                scale_s_nuc[..., 0],
                0.1 * scale_s_nuc[..., 0],
            ),
            dim=2,
        )
        dist_s_nuc = Normal(mean_s_nuc, scale_s_nuc)
        mixture_dist_s_nuc = MixtureSameFamily(comp_dist, dist_s_nuc)

        # spliced cytoplasm
        scale_s_cyt = torch.stack(
            (
                scale_s_cyt[..., 0],
                scale_s_cyt[..., 0],
                scale_s_cyt[..., 0],
                0.1 * scale_s_cyt[..., 0],
            ),
            dim=2,
        )
        mean_s_cyt = torch.stack(
            (
                mean_s_cyt_ind,
                mean_s_cyt_ind_steady,
                mean_s_cyt_rep,
                mean_s_cyt_rep_steady,
            ),
            dim=2,
        )
        dist_s_cyt = Normal(mean_s_cyt, scale_s_cyt)
        mixture_dist_s_cyt = MixtureSameFamily(comp_dist, dist_s_cyt)

        return mixture_dist_s_cyt, mixture_dist_s_nuc, mixture_dist_u_nuc, end_penalty

    def _get_induction_steady_states(self, alpha, beta, nu, gamma, n_cells, eps=1e-6):
        mean_u_nuc_ind_steady = (alpha / beta).expand(n_cells, self.n_input)
        mean_s_nuc_ind_steady = (alpha / nu).expand(n_cells, self.n_input)
        mean_s_cyt_ind_steady = (
            nu
            * beta
            * (
                (
                    (alpha / beta) * (gamma - nu)
                    - (alpha / nu) * (gamma - beta)
                    + (alpha / gamma) * (nu - beta)
                )
                / (((gamma - nu) * (gamma - beta) * (nu - beta)) + eps)
            )
        ).expand(n_cells, self.n_input)

        return mean_u_nuc_ind_steady, mean_s_nuc_ind_steady, mean_s_cyt_ind_steady

    def _get_induction_unspliced_spliced(self, alpha, beta, nu, gamma, t, eps=1e-6):
        unspliced_nuc = (alpha / beta) * (1 - torch.exp(-beta * t))

        spliced_nuc = (alpha / nu) * (1 - torch.exp(-nu * t)) + (
            alpha / ((nu - beta) + eps)
        ) * (torch.exp(-nu * t) - torch.exp(-beta * t))

        spliced_cyt = (
            nu
            * beta
            * (
                alpha
                / beta
                * (1 - torch.exp(-beta * t))
                / (((nu - beta) * (gamma - beta)) + eps)
                - alpha
                / nu
                * (1 - torch.exp(-nu * t))
                / (((gamma - nu) * (nu - beta)) + eps)
                + alpha
                / gamma
                * (1 - torch.exp(-gamma * t))
                / (((gamma - nu) * (gamma - beta)) + eps)
            )
        )

        return unspliced_nuc, spliced_nuc, spliced_cyt

    def _get_repression_unspliced_spliced(
        self, u_nuc_0, s_nuc_0, s_cyt_0, beta, nu, gamma, t, eps=1e-6
    ):
        unspliced_nuc = u_nuc_0 * torch.exp(-beta * t)

        spliced_nuc = s_nuc_0 * torch.exp(-nu * t) - (
            (beta * u_nuc_0) / (nu - beta + eps)
        ) * (torch.exp(-nu * t) - torch.exp(-beta * t))

        spliced_cyt = (
            nu
            * beta
            * u_nuc_0
            * (
                (torch.exp(-beta * t)) / (((nu - beta) * (gamma - beta)) + eps)
                - (torch.exp(-nu * t)) / (((nu - beta) * (gamma - nu)) + eps)
                + (torch.exp(-gamma * t)) / (((gamma - nu) * (gamma - beta)) + eps)
            )
            + (nu * s_nuc_0)
            / (gamma - nu + eps)
            * (torch.exp(-nu * t) - torch.exp(-gamma * t))
            + s_cyt_0 * torch.exp(-gamma * t)
        )

        return unspliced_nuc, spliced_nuc, spliced_cyt

    def sample(self) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings
