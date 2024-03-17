import warnings
from typing import List, Optional, Union

import numpy as np

from anndata import AnnData

from scvelo.core import invert, SplicingDynamics


# TODO: Add docstrings
# TODO Use `SplicingDynamics`
def unspliced(tau, u0, alpha, beta):
    """TODO."""
    expu = np.exp(-beta * tau)
    return u0 * expu + alpha / beta * (1 - expu)


# TODO: Add docstrings
def spliced(tau, s0, u0, alpha, beta, gamma):
    """TODO."""
    c = (alpha - u0 * beta) * invert(gamma - beta)
    expu, exps = np.exp(-beta * tau), np.exp(-gamma * tau)
    return s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu)


# TODO: Add docstrings
def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0, sorted=False):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)
    s0_ = spliced(t_, s0, u0, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, s0


def simulation(
    n_obs=300,
    n_vars=None,
    alpha=None,
    beta=None,
    gamma=None,
    alpha_=None,
    t_max=None,
    noise_model="normal",
    noise_level=1,
    switches=None,
    random_seed=0,
):
    """Simulation of mRNA splicing kinetics.

    Simulated mRNA metabolism with transcription, splicing and degradation.
    The parameters for each reaction are randomly sampled from a log-normal distribution
    and time events follow the Poisson law. The total time spent in a transcriptional
    state is varied between two and ten hours.

    .. image:: https://user-images.githubusercontent.com/31883718/79432471-16c0a000-7fcc-11ea-8d62-6971bcf4181a.png
       :width: 600px

    Returns
    -------
    Returns `adata` object
    """
    np.random.seed(random_seed)

    def draw_poisson(n):
        from random import seed, uniform  # draw from poisson

        seed(random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # prepend t0=0

    def simulate_dynamics(tau, alpha, beta, gamma, u0, s0, noise_model, noise_level):
        ut, st = SplicingDynamics(
            transcription_rate=alpha,
            splicing_rate=beta,
            degradation_rate=gamma,
            initial_state=[u0, s0],
        ).get_solution(tau, stacked=False)
        if noise_model == "normal":  # add noise
            ut += np.random.normal(
                scale=noise_level * np.percentile(ut, 99) / 10, size=len(ut)
            )
            st += np.random.normal(
                scale=noise_level * np.percentile(st, 99) / 10, size=len(st)
            )
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        return ut, st

    def simulate_gillespie(alpha, beta, gamma):
        # update rules:
        # transcription (u+1,s), splicing (u-1,s+1), degradation (u,s-1), nothing (u,s)
        update_rule = np.array([[1, 0], [-1, 1], [0, -1], [0, 0]])

        def update(props):
            if np.sum(props) > 0:
                props /= np.sum(props)
            p_cumsum = props.cumsum()
            p = np.random.rand()
            i = 0
            while p > p_cumsum[i]:
                i += 1
            return update_rule[i]

        u, s = np.zeros(len(alpha)), np.zeros(len(alpha))
        for i, alpha_i in enumerate(alpha):
            u_, s_ = (u[i - 1], s[i - 1]) if i > 0 else (0, 0)

            if (alpha_i == 0) and (u_ == 0) and (s_ == 0):
                du, ds = 0, 0
            else:
                du, ds = update(props=np.array([alpha_i, beta * u_, gamma * s_]))

            u[i], s[i] = (u_ + du, s_ + ds)
        return u, s

    alpha = 5 if alpha is None else alpha
    beta = 0.5 if beta is None else beta
    gamma = 0.3 if gamma is None else gamma
    alpha_ = 0 if alpha_ is None else alpha_

    t = draw_poisson(n_obs)
    if t_max is not None:
        t *= t_max / np.max(t)
    t_max = np.max(t)

    def cycle(array, n_vars=None):
        if isinstance(array, (np.ndarray, list, tuple)):
            return (
                array if n_vars is None else array * int(np.ceil(n_vars / len(array)))
            )
        else:
            return [array] if n_vars is None else [array] * n_vars

    # switching time point obtained as fraction of t_max rounded down
    switches = (
        cycle([0.4, 0.7, 1, 0.1], n_vars)
        if switches is None
        else cycle(switches, n_vars)
    )
    t_ = np.array([np.max(t[t < t_i * t_max]) for t_i in switches])

    noise_level = cycle(noise_level, len(switches) if n_vars is None else n_vars)

    n_vars = min(len(switches), len(noise_level)) if n_vars is None else n_vars
    U = np.zeros(shape=(len(t), n_vars))
    S = np.zeros(shape=(len(t), n_vars))

    def is_list(x):
        return isinstance(x, (tuple, list, np.ndarray))

    for i in range(n_vars):
        alpha_i = alpha[i] if is_list(alpha) and len(alpha) != n_obs else alpha
        beta_i = beta[i] if is_list(beta) and len(beta) != n_obs else beta
        gamma_i = gamma[i] if is_list(gamma) and len(gamma) != n_obs else gamma
        tau, alpha_vec, u0_vec, s0_vec = vectorize(
            t, t_[i], alpha_i, beta_i, gamma_i, alpha_=alpha_, u0=0, s0=0
        )

        if noise_model == "gillespie":
            U[:, i], S[:, i] = simulate_gillespie(alpha_vec, beta, gamma)
        else:
            U[:, i], S[:, i] = simulate_dynamics(
                tau,
                alpha_vec,
                beta_i,
                gamma_i,
                u0_vec,
                s0_vec,
                noise_model,
                noise_level[i],
            )

    if is_list(alpha) and len(alpha) == n_obs:
        alpha = np.nan
    if is_list(beta) and len(beta) == n_obs:
        beta = np.nan
    if is_list(gamma) and len(gamma) == n_obs:
        gamma = np.nan

    obs = {"true_t": t.round(2)}
    var = {
        "true_t_": t_[:n_vars],
        "true_alpha": np.ones(n_vars) * alpha,
        "true_beta": np.ones(n_vars) * beta,
        "true_gamma": np.ones(n_vars) * gamma,
        "true_scaling": np.ones(n_vars),
    }
    layers = {"unspliced": U, "spliced": S}

    return AnnData(S, obs, var, layers=layers)


# TODO: Check for better location of function definition
def broadcast(a, n: Optional[int] = None):
    """TODO."""
    if isinstance(a, (np.ndarray, list, tuple)):
        return a if n is None else a * int(np.ceil(n / len(a)))
    else:
        return [a] if n is None else [a] * n


# TODO make backward compatible by wrapping `Simulator` with `simulation`
# TODO: Update type hints: switches, t0, t_final must be positive floats
# TODO: Add noise model gillespie
class Simulator:
    def __init__(
        self,
        n_obs: int = 300,
        n_vars: int = 1000,
        noise: str = "normal",
        noise_level: Union[List[float], float] = 0.5,
        switches: Optional[Union[List[float], float]] = None,
        random_seed: int = 0,
        t0: float = 0,
        t_final: Optional[float] = None,
        cell_gene_time: bool = False,
        time_distribution: str = "uniform",
    ):
        """Class for ODE-based data simulation.

        Arguments
        ---------
        n_obs
            Number of cells.
        n_vars
            Number of genes.
        noise
            if == "normal" then normal noise is added to abundances
        noise_level
            Variance of added noise
        switches
            Switching times for genes
        random_seed
            Seed.
        t0
            Minimum t (default to 0). Needs to be >= 0
        t_final
            Maximum t. Needs to be >= 0
        cell_gene_time
            Wether to sample cell-gene times e.g. n_obs x n_vars time points or just one latent time per cell e.g. of shape (n_obs,)
        time_distribution
            If specified as "uniform" and `cell_gene_time=True´ the cell-gene times are sampled from a uniform distribution on [0,1] s.t. each newly sampled time gets added within a running sum to ensure that we have one cell for approx. each time step. Afterwards the times for each gene are randomly shuffled.

        Examples
        --------
        Splicing dynamics can be simulated as follows:


        For cell_gene_time=True and NucCytModel:
        >>> from scvelo.core import SplicingDynamics, NucCytModel
        >>> from scvelo.data import Simulator
        >>>
        >>> simulator = Simulator(cell_gene_time=True)
        >>> adata = simulator.fit(
        >>>     NucCytModel,
        >>>     layer_names=["U_NUC", "S_NUC", "S_CYT"],
        >>>     transcription_rate=5,
        >>>     splicing_rate=0.5,
        >>>     nuc_export_rate=0.25
        >>>     degradation_rate=0.1,
        >>>     transcription_rate_=0,
        >>>     initial_state=[0, 0, 0]
        >>> )
        Also possible to pass rate parameters as np.array of shape (n_vars,)

        For cell_gene_time=False and SplicingDynamics
        >>> simulator = Simulator(cell_gene_time=False)
        >>> adata = simulator.fit(
        >>>     SplicingDynamics,
        >>>     layer_names=["unspliced", "spliced"],
        >>>     transcription_rate=5,
        >>>     splicing_rate=0.5,
        >>>     degradation_rate=0.1,
        >>>     transcription_rate_=0,
        >>>     initial_state=[0, 0]
        >>> )
        """
        self.random_seed = random_seed
        self.n_vars = n_vars
        self.n_obs = n_obs
        self.t0 = t0
        self.t_final = t_final

        np.random.seed(self.random_seed)

        if cell_gene_time:
            if time_distribution == "poisson":
                self.t = self.get_poisson_time()
            else:
                self.t = self.get_uniform_time()
        else:
            if n_obs == 1:
                if t_final is None:
                    self.t_final = t0 + 1
                self.t = np.array([t0, t_final])
            else:
                self.t = np.insert(
                    np.cumsum(-0.1 * np.log(np.random.uniform(size=n_obs - 1))), 0, 0
                )

                if t_final is None:
                    self.t_final = self.t.max()
                self.t = t0 + self.t * (self.t_final - t0) / self.t.max()

        self.switches = broadcast(
            [0.1, 0.4, 0.7, 1] if switches is None else switches, n_vars
        )

        self.switching_times = np.array(
            [
                np.max(self.t[self.t < t0 + switching_time * self.t_final])
                for switching_time in self.switches
            ]
        )

        self.noise = noise
        self.noise_level = broadcast(
            noise_level, len(self.switches) if n_vars is None else n_vars
        )

        self.n_vars = (
            min(len(self.switches), len(self.noise_level)) if n_vars is None else n_vars
        )

    def get_uniform_time(self):
        """Function draws uniform distributed cell-gene times and uses a running sum, s.t. each cell has a different time. Afterwards scales the values to [t_min, t_max]."""
        t = np.zeros((self.n_obs, self.n_vars))
        for gene in range(self.n_vars):
            cell_times = np.insert(
                np.cumsum(-0.1 * np.log(np.random.uniform(size=self.n_obs - 1))), 0, 0
            )
            np.random.shuffle(cell_times)
            if self.t_final is None:
                self.t_final = cell_times.max()
            cell_times = (
                self.t0 + cell_times * (self.t_final - self.t0) / cell_times.max()
            )
            t[:, gene] = cell_times

        return t

    def get_poisson_time(self):
        """Function draws poisson distributed cell-gene times and scales the values to [t_min, t_max]."""
        # set lambda parameter to 10, shuffle sample and scale values afterwards
        t_poisson = np.random.poisson(10, self.n_vars * self.n_obs)
        np.random.shuffle(t_poisson)
        t_poisson = t_poisson.reshape((self.n_obs, self.n_vars))

        # MinMax Scale times s.t. each gene has a cell with t_ig=t_min and t_ig=t_max
        if self.t_final is None:
            self.t_final = t_poisson.max()
        t_std = (t_poisson - t_poisson.min(axis=0)) / (
            t_poisson.max(axis=0) - t_poisson.min(axis=0)
        )
        t_poisson = t_std * (self.t_final - self.t0) + self.t0

        return t_poisson

    def _add_noise(self, sol, noise_level):
        """Function adds gaussian noise to array.

        Parameters
        ----------
        sol
            Transcript abundances (u_nuc,s_nuc,s_cyt)
        noise_model
            Whether to add normal noise to abundances (if ==normal, else return sol)
        noise_level
        Scale factor for noise.
        """
        np.random.seed(self.random_seed)
        if self.noise == "normal":
            return np.clip(
                sol
                + np.random.normal(
                    scale=noise_level * np.percentile(sol, 99, axis=0) / 10,
                    size=sol.shape,
                ),
                0,
                None,
            ).T
        else:
            return sol.T

    def get_time_update(self, switching_time, var_id):
        """Function returns tau (e.g. t_ig - t_0) depending on induction/ repression state.

        Parameters
        ----------
        switching_time
            Switching times of all genes with shape (self.n_genes,)
        var_id
            Id of gene, which is needed to subsample time of all cells of particular gene, if time is of shape (n_obs,n_vars)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.t.ndim == 1:
                o = np.array(self.t < switching_time, dtype=int)
                return self.t * o + (self.t - switching_time) * (1 - o)
            else:
                o = np.array(self.t[:, var_id] < switching_time, dtype=int)
                return self.t[:, var_id] * o + (self.t[:, var_id] - switching_time) * (
                    1 - o
                )

    def _update_and_vectorize_parameters(
        self, switching_time, dynamics, var_id, **params
    ):
        """Function vectorizes rate parameters as well as initial states per gene. Can be used for transcription_rate(_) which is set to 0 for repression phase.

        Parameters
        ----------
        switching_time
            Switching times of gene with var_id
        dynamics
            Instance of scvelo/core/_models (e.g. NucCytModel/ SplicingDynamics)
        var_id
            Id of gene, which is needed to subsample time of all cells of particular gene, if time is of shape (n_obs,n_vars)
        **params: Possible kwargs to specify, One can also specify ..._rate_ to have state dependent rates
            initial_state
                Initial state of system. Defaults to `[0, 0, 0]`. correpsonding to u0_nuc, s0_nuc, s0_cyt
            transcription_rate
                Transcription rate of gene g (e.g. a float) in induction state
            transcription_rate_ (=0)
                Transcription rate of gene g (e.g. a float) in repression phase
            splicing_rate
                Splicing/ Translation rate.
            nuc_export_rate
                Nuclear export rate to cytoplasm.
            degradation_rate
                Splicing degradation rate.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.t.ndim == 1:
                o = np.array(self.t < switching_time, dtype=int)
            else:
                o = np.array(self.t[:, var_id] < switching_time, dtype=int)

        parameters_before_split = [key for key in params if not key.endswith("_")]

        states_after_switch = dynamics.get_solution(switching_time)

        for parameter in parameters_before_split:
            if parameter == "initial_state":
                dynamics.initial_state = np.array(params["initial_state"]).reshape(
                    1, -1
                ) * o.reshape(-1, 1) + states_after_switch.reshape(1, -1) * (
                    1 - o
                ).reshape(
                    -1, 1
                )
            else:
                setattr(
                    dynamics,
                    parameter,
                    params[parameter] * o
                    + params.get(f"{parameter}_", params[parameter]) * (1 - o),
                )

    def fit(
        self,
        ode_class,
        layer_names: List,
        X_name: Optional[str] = None,
        **params,
    ):
        """Function returns adata object with simulated data based on specified parameters and Splicing Dynamics.

        Parameters
        ----------
        dynamics
            Name of scvelo/core/_models (e.g. NucCytModel/ SplicingDynamics)
        ode_class
            Names of anndata layers, for SplicingDynamics e.g. ["unspliced", "spliced"] and for NucCytModel ["U_NUC", "S_NUC", "S_CYT"]
        X_name
            Name of layer to store as adata.X
        **params: Possible kwargs to specify, One can also specify ..._rate_ to have state dependent rates
            initial_state
                Initial state of system. Set e.g. to `[0, 0]` for SplicingDynamics or to `[0, 0, 0]` for NucCytModel correpsonding to u0_nuc, s0_nuc, s0_cyt
            transcription_rate
                Transcription rate of gene g (e.g. a float) in induction state.
            transcription_rate_ (=0)
                Transcription rate of gene g (e.g. a float) in repression phase.
            splicing_rate
                Splicing/ Translation rate.
            nuc_export_rate
                Nuclear export rate to cytoplasm (required for NucCytModel, and not required for SplicingDynamics).
            degradation_rate
                Splicing degradation rate.
        """
        states = np.zeros(shape=(len(layer_names), len(self.t), self.n_vars))

        initial_states = params.pop("initial_state")
        for var_id in range(self.n_vars):
            _params = {
                key: val[var_id] if isinstance(val, (tuple, list, np.ndarray)) else val
                for key, val in params.items()
            }

            if isinstance(initial_states[0], list):
                _params.update({"initial_state": initial_states[var_id]})
            else:
                _params.update({"initial_state": initial_states})

            dynamics = ode_class(
                **{key: val for key, val in _params.items() if not key.endswith("_")}
            )

            tau = self.get_time_update(self.switching_times[var_id], var_id)
            self._update_and_vectorize_parameters(
                self.switching_times[var_id], dynamics, var_id, **_params
            )

            sol = dynamics.get_solution(tau)

            states[:, :, var_id] = self._add_noise(
                sol=sol, noise_level=self.noise_level[var_id]
            )

        layers = {
            layer_names[n_dim]: states[n_dim, :, :] for n_dim in range(states.shape[0])
        }
        # if we have cell-gene times, store true_t as layer, else as obs
        if self.t.ndim == 1:
            obs = {"true_t": self.t.round(2)}
        else:
            obs = {}
            layers["true_t"] = self.t.round(2)
        var = {
            "true_t_": self.switching_times[: self.n_vars],
            "true_scaling": np.ones(self.n_vars),
        }
        var.update(
            {f"true_{key}": val for key, val in params.items() if not key.endswith("_")}
        )

        return AnnData(
            states[0, :, :] if X_name is None else layers[X_name],
            obs,
            var,
            layers=layers,
        )
