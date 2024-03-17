import mplscience

import numpy as np
import pandas as pd

import matplotlib.pyplot as pl
import seaborn as sns
from matplotlib import rcParams

from scvelo.core import NucCytModel, SplicingDynamics
from scvelo.tools._em_model_utils import get_vars, tau_inv, unspliced, vectorize
from .utils import make_dense


# TODO: Add docstrings
def get_dynamics(adata, key="fit", extrapolate=False, sorted=False, t=None):
    """TODO."""
    alpha, beta, gamma, scaling, t_ = get_vars(adata, key=key)
    if extrapolate:
        u0_ = unspliced(t_, 0, alpha, beta)
        tmax = t_ + tau_inv(u0_ * 1e-4, u0=u0_, alpha=0, beta=beta)
        t = np.concatenate(
            [np.linspace(0, t_, num=500), t_ + np.linspace(0, tmax, num=500)]
        )
    elif t is None or t is True:
        t = adata.obs[f"{key}_t"].values if key == "true" else adata.layers[f"{key}_t"]

    tau, alpha, u0, s0 = vectorize(np.sort(t) if sorted else t, t_, alpha, beta, gamma)
    ut, st = SplicingDynamics(
        transcription_rate=alpha,
        splicing_rate=beta,
        degradation_rate=gamma,
        initial_state=[u0, s0],
    ).get_solution(tau)
    return alpha, ut, st


# TODO: Add docstrings
def compute_dynamics(
    adata, basis, key="true", extrapolate=None, sort=True, t_=None, t=None
):
    """TODO."""
    idx = adata.var_names.get_loc(basis) if isinstance(basis, str) else basis
    key = "fit" if f"{key}_gamma" not in adata.var_keys() else key
    alpha, beta, gamma, scaling, t_ = get_vars(adata[:, basis], key=key)

    if "fit_u0" in adata.var.keys():
        u0_offset, s0_offset = adata.var["fit_u0"][idx], adata.var["fit_s0"][idx]
    else:
        u0_offset, s0_offset = 0, 0

    if t is None or isinstance(t, bool) or len(t) < adata.n_obs:
        t = (
            adata.obs[f"{key}_t"].values
            if key == "true"
            else adata.layers[f"{key}_t"][:, idx]
        )

    if extrapolate:
        u0_ = unspliced(t_, 0, alpha, beta)
        tmax = np.max(t) if True else tau_inv(u0_ * 1e-4, u0=u0_, alpha=0, beta=beta)
        t = np.concatenate(
            [np.linspace(0, t_, num=500), np.linspace(t_, tmax, num=500)]
        )

    tau, alpha, u0, s0 = vectorize(np.sort(t) if sort else t, t_, alpha, beta, gamma)

    ut, st = SplicingDynamics(
        transcription_rate=alpha,
        splicing_rate=beta,
        degradation_rate=gamma,
        initial_state=[u0, s0],
    ).get_solution(tau, stacked=False)
    ut, st = ut * scaling + u0_offset, st + s0_offset
    return alpha, ut, st


# TODO: Add docstrings
def show_full_dynamics(
    adata,
    basis,
    key="true",
    use_raw=False,
    linewidth=1,
    linecolor=None,
    show_assignments=None,
    ax=None,
):
    """TODO."""
    if ax is None:
        ax = pl.gca()
    color = linecolor if linecolor else "grey" if key == "true" else "purple"
    linewidth = 0.5 * linewidth if key == "true" else linewidth
    label = "learned dynamics" if key == "fit" else "true dynamics"
    line = None

    if key != "true":
        _, ut, st = compute_dynamics(
            adata, basis, key, extrapolate=False, sort=False, t=show_assignments
        )
        if not isinstance(show_assignments, str) or show_assignments != "only":
            ax.scatter(st, ut, color=color, s=1)
        if show_assignments is not None and show_assignments is not False:
            skey, ukey = (
                ("spliced", "unspliced")
                if use_raw or "Ms" not in adata.layers.keys()
                else ("Ms", "Mu")
            )
            s, u = (
                make_dense(adata[:, basis].layers[skey]).flatten(),
                make_dense(adata[:, basis].layers[ukey]).flatten(),
            )
            ax.plot(
                np.array([s, st]),
                np.array([u, ut]),
                color="grey",
                linewidth=0.1 * linewidth,
            )

    if not isinstance(show_assignments, str) or show_assignments != "only":
        _, ut, st = compute_dynamics(
            adata, basis, key, extrapolate=True, t=show_assignments
        )
        (line,) = ax.plot(st, ut, color=color, linewidth=linewidth, label=label)

        idx = adata.var_names.get_loc(basis)
        beta, gamma = adata.var[f"{key}_beta"][idx], adata.var[f"{key}_gamma"][idx]
        xnew = np.linspace(np.min(st), np.max(st))
        ynew = gamma / beta * (xnew - np.min(xnew)) + np.min(ut)
        ax.plot(xnew, ynew, color=color, linestyle="--", linewidth=linewidth)
    return line, label


# TODO: Add docstrings
def simulation(
    adata,
    var_names="all",
    legend_loc="upper right",
    legend_fontsize=20,
    linewidth=None,
    dpi=None,
    xkey="true_t",
    ykey=None,
    colors=None,
    **kwargs,
):
    """TODO."""
    from scvelo.tools.utils import make_dense
    from .scatter import scatter

    if ykey is None:
        ykey = ["unspliced", "spliced", "alpha"]
    if colors is None:
        colors = ["darkblue", "darkgreen", "grey"]
    var_names = (
        adata.var_names
        if isinstance(var_names, str) and var_names == "all"
        else [name for name in var_names if name in adata.var_names]
    )

    figsize = rcParams["figure.figsize"]
    ncols = len(var_names)
    for i, gs in enumerate(
        pl.GridSpec(
            1, ncols, pl.figure(None, (figsize[0] * ncols, figsize[1]), dpi=dpi)
        )
    ):
        idx = adata.var_names.get_loc(var_names[i])
        alpha, ut, st = compute_dynamics(adata, idx)
        t = (
            adata.obs[xkey]
            if xkey in adata.obs.keys()
            else make_dense(adata.layers["fit_t"][:, idx])
        )
        idx_sorted = np.argsort(t)
        t = t[idx_sorted]

        ax = pl.subplot(gs)
        _kwargs = {"alpha": 0.3, "title": "", "xlabel": "time", "ylabel": "counts"}
        _kwargs.update(kwargs)
        linewidth = 1 if linewidth is None else linewidth

        ykey = [ykey] if isinstance(ykey, str) else ykey
        for j, key in enumerate(ykey):
            if key in adata.layers:
                y = make_dense(adata.layers[key][:, idx])[idx_sorted]
                ax = scatter(x=t, y=y, color=colors[j], ax=ax, show=False, **_kwargs)

            if key == "unspliced":
                ax.plot(t, ut, label="unspliced", color=colors[j], linewidth=linewidth)
            elif key == "spliced":
                ax.plot(t, st, label="spliced", color=colors[j], linewidth=linewidth)
            elif key == "alpha":
                largs = {"linewidth": linewidth, "linestyle": "--"}
                ax.plot(t, alpha, label="alpha", color=colors[j], **largs)

        pl.xlim(0)
        pl.ylim(0)
        if legend_loc != "none" and i == ncols - 1:
            pl.legend(loc=legend_loc, fontsize=legend_fontsize)


def get_nuc_cyt_dynamics(adata, gene, extrapolate=True):
    """Get fitted nuc/cyt dynamics of specific gene.

    Parameters
    ----------
    adata
        Annotated data object.
    gene
        String of name of gene
    extrapolate
        Whether to extrapolate cell times
    """
    subset_adata = adata[:, gene]
    alpha, beta, nu, gamma = (
        subset_adata.var["fit_alpha"],
        subset_adata.var["fit_beta"],
        subset_adata.var["fit_nu"],
        subset_adata.var["fit_gamma"],
    )
    cell_t = subset_adata.layers["fit_t"]
    switching_time = subset_adata.var["fit_t_"]

    # all to array
    alpha = np.array(alpha)
    beta = np.array(beta)
    nu = np.array(nu)
    gamma = np.array(gamma)
    cell_t = np.array(cell_t)
    switching_time = np.array(switching_time)

    o = np.array(cell_t < switching_time, dtype=int)

    if extrapolate:
        # u0_ = unspliced(np.array(switching_time), 0, np.array(alpha), np.array(beta))
        tmax = np.max(
            cell_t
        )  # if True else tau_inv(u0_ * 1e-4, u0=u0_, alpha=0, beta=beta)
        cell_t = np.concatenate(
            [
                np.linspace(0, switching_time, num=500),
                np.linspace(switching_time, tmax, num=500),
            ]
        )

    o = np.array(cell_t < switching_time, dtype=int)

    u0_nuc, s0_nuc, s0_cyt = [0, 0, 0]
    tau = cell_t * o + (cell_t - switching_time) * (1 - o)
    # get initial condition for repression phase
    u0_nuc_, s0_nuc_, s0_cyt_ = NucCytModel(
        transcription_rate=alpha,
        splicing_rate=beta,
        nuc_export_rate=nu,
        degradation_rate=gamma,
        initial_state=[u0_nuc, s0_nuc, s0_cyt],
    ).get_solution(switching_time, stacked=False)

    u0_nuc = u0_nuc * o + u0_nuc_ * (1 - o)
    s0_nuc = s0_nuc * o + s0_nuc_ * (1 - o)
    s0_cyt = s0_cyt * o + s0_cyt_ * (1 - o)
    alpha_ = 0
    alpha = alpha * o + alpha_ * (1 - o)

    NucCytModel(
        transcription_rate=alpha,
        splicing_rate=beta,
        nuc_export_rate=nu,
        degradation_rate=gamma,
        initial_state=[u0_nuc, s0_nuc, s0_cyt],
    )

    ut_nuc, st_nuc, st_cyt = NucCytModel(
        transcription_rate=alpha,
        splicing_rate=beta,
        nuc_export_rate=nu,
        degradation_rate=gamma,
        initial_state=[u0_nuc, s0_nuc, s0_cyt],
    ).get_solution(tau, stacked=False)

    ut_nuc_steady, st_nuc_steady, st_cyt_steady = NucCytModel(
        transcription_rate=np.array(subset_adata.var["fit_alpha"]),
        splicing_rate=np.array(subset_adata.var["fit_beta"]),
        nuc_export_rate=np.array(subset_adata.var["fit_nu"]),
        degradation_rate=np.array(subset_adata.var["fit_gamma"]),
        initial_state=[u0_nuc, s0_nuc, s0_cyt],
    ).get_steady_states(stacked=False)

    ut_nuc_steady = np.unique(ut_nuc_steady)
    st_nuc_steady = np.unique(st_nuc_steady)
    st_cyt_steady = np.unique(st_cyt_steady)

    return ut_nuc, st_nuc, st_cyt, ut_nuc_steady, st_nuc_steady, st_cyt_steady


def plot_phase_portrait_nuc_cyt(
    adata, gene, color, layer_x="Ms_cyt", layer_y="Ms_nuc", figsize=(6, 6)
):
    """Plot Phase portrait of layer_x vs layer_y.

    Function also plots fitted Nuc/cyt dynamics including steady state line.

    Parameters
    ----------
    adata
        Annotated data object.
    gene
        String of name of gene
    color
        color
    layer_x
        needs to be present in `adata.layers`
    layer_y
        needs to be present in `adata.layers`
    figsize
        Size of figure
    """
    fig, ax = pl.subplots(figsize=figsize)

    df = pd.DataFrame(
        {
            layer_x: adata[:, gene].layers[layer_x].toarray().squeeze(),
            layer_y: adata[:, gene].layers[layer_y].toarray().squeeze(),
            "color": color,
        }
    )

    with mplscience.style_context():
        sns.scatterplot(data=df, x=layer_x, y=layer_y, c=color, s=25, ax=ax)

        (
            ut_nuc,
            st_nuc,
            st_cyt,
            ut_nuc_steady,
            st_nuc_steady,
            st_cyt_steady,
        ) = get_nuc_cyt_dynamics(adata, gene, extrapolate=True)
        map = {"Mu_nuc": ut_nuc, "Ms_nuc": st_nuc, "Ms_cyt": st_cyt}
        steady_range_x = np.linspace(np.min(map[layer_x]), np.max(map[layer_x]))
        u_nuc_s_nuc_steady = ut_nuc_steady / st_nuc_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])
        u_nuc_s_cyt_steady = ut_nuc_steady / st_cyt_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])
        s_nuc_s_cyt_steady = st_nuc_steady / st_cyt_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])

        steady_state_map = {
            "Mu_nuc_Ms_nuc": u_nuc_s_nuc_steady,
            "Mu_nuc_Ms_cyt": u_nuc_s_cyt_steady,
            "Ms_nuc_Ms_cyt": s_nuc_s_cyt_steady,
        }
        ax.plot(map[layer_x], map[layer_y], color="purple", linewidth=2)
        ax.plot(
            steady_range_x,
            steady_state_map[layer_y + "_" + layer_x],
            color="purple",
            linestyle="--",
            linewidth=2,
        )

    # ax.axis('off')


def plot_nuc_cyt_dynamics(
    adata, gene, ax, color="purple", layer_x="Ms_cyt", layer_y="Ms_nuc"
):
    """Plot only inferred dynamics of Nuc/cyt model.

    Function also plots fitted Nuc/cyt dynamics including steady state line.



    Parameters
    ----------
    adata
        Annotated data object.
    gene
        String of name of gene
    color
        color
    ax
        Axis for ax.plot
    layer_x
        needs to be present in `adata.layers`
    layer_y
        needs to be present in `adata.layers`

    """
    with mplscience.style_context():
        (
            ut_nuc,
            st_nuc,
            st_cyt,
            ut_nuc_steady,
            st_nuc_steady,
            st_cyt_steady,
        ) = get_nuc_cyt_dynamics(adata, gene, extrapolate=True)
        map = {"Mu_nuc": ut_nuc, "Ms_nuc": st_nuc, "Ms_cyt": st_cyt}
        steady_range_x = np.linspace(np.min(map[layer_x]), np.max(map[layer_x]))
        u_nuc_s_nuc_steady = ut_nuc_steady / st_nuc_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])
        u_nuc_s_cyt_steady = ut_nuc_steady / st_cyt_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])
        s_nuc_s_cyt_steady = st_nuc_steady / st_cyt_steady * (
            steady_range_x - np.min(steady_range_x)
        ) + np.min(map[layer_y])

        steady_state_map = {
            "Mu_nuc_Ms_nuc": u_nuc_s_nuc_steady,
            "Mu_nuc_Ms_cyt": u_nuc_s_cyt_steady,
            "Ms_nuc_Ms_cyt": s_nuc_s_cyt_steady,
        }
        ax.plot(map[layer_x], map[layer_y], color=color, linewidth=2)
        ax.plot(
            steady_range_x,
            steady_state_map[layer_y + "_" + layer_x],
            color=color,
            linestyle="--",
            linewidth=2,
        )
