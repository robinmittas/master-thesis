from typing import Dict, List, Tuple, Union

import numpy as np
from numpy import ndarray

from ._arithmetic import invert
from ._base import DynamicsBase


# TODO: Handle cases splicing_rate = 0, degradation_rate == 0, splicing_rate == degradation_rate
class SplicingDynamics(DynamicsBase):
    """Splicing dynamics.

    Arguments
    ---------
    transcription_rate
        Transcription rate.
    splicing_rate
        Translation rate.
    degradation_rate
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0]`.

    Attributes
    ----------
    transcription_rate
        Transcription rate.
    splicing_rate
        Translation rate.
    degradation_rate
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0]`.
    u0
        Initial abundance of unspliced RNA.
    s0
        Initial abundance of spliced RNA.

    """

    def __init__(
        self,
        transcription_rate: float,
        splicing_rate: float,
        degradation_rate: float,
        initial_state: Union[List, ndarray] = None,
    ):
        self.transcription_rate = transcription_rate
        self.splicing_rate = splicing_rate
        self.degradation_rate = degradation_rate

        if initial_state is None:
            self.initial_state = [0, 0]
        else:
            self.initial_state = initial_state

    @property
    def initial_state(self):
        """Function to get initial values as list [u0, s0]."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        """Sets the initial values of undpliced and spliced abundances.

        Arguments:
        ---------
        val
            containing [u0,s0].
        """
        if isinstance(val, list) or (isinstance(val, ndarray) and (val.ndim == 1)):
            self.u0 = val[0]
            self.s0 = val[1]
        else:
            self.u0 = val[:, 0]
            self.s0 = val[:, 1]
        self._initial_state = val

    def get_solution(
        self, t: ndarray, stacked: bool = True, with_keys: bool = False
    ) -> Union[Dict, ndarray]:
        """Calculate solution of dynamics.

        Arguments
        ---------
        t
            Time steps at which to evaluate solution.
        stacked
            Whether to stack states or return them individually. Defaults to `True`.
        with_keys
            Whether to return solution labelled by variables in form of a dictionary.
            Defaults to `False`.

        Returns
        -------
        Union[Dict, ndarray]
            Solution of system. If `with_keys=True`, the solution is returned in form of
            a dictionary with variables as keys. Otherwise, the solution is given as
            a `numpy.ndarray` of form `(n_steps, 2)`.
        """
        expu = np.exp(-self.splicing_rate * t)
        exps = np.exp(-self.degradation_rate * t)

        unspliced = self.u0 * expu + self.transcription_rate / self.splicing_rate * (
            1 - expu
        )
        c = (self.transcription_rate - self.u0 * self.splicing_rate) * invert(
            self.degradation_rate - self.splicing_rate
        )
        spliced = (
            self.s0 * exps
            + self.transcription_rate / self.degradation_rate * (1 - exps)
            + c * (exps - expu)
        )

        if with_keys:
            return {"u": unspliced, "s": spliced}
        elif not stacked:
            return unspliced, spliced
        else:
            if isinstance(t, np.ndarray) and t.ndim == 2:
                return np.stack([unspliced, spliced], axis=2)
            else:
                return np.column_stack([unspliced, spliced])

    # TODO: Handle cases `splicing_rate = 0`, `degradation_rate = 0`
    def get_steady_states(
        self, stacked: bool = True, with_keys: bool = False
    ) -> Union[Dict[str, ndarray], Tuple[ndarray], ndarray]:
        """Return steady state of system.

        Arguments
        ---------
        stacked
            Whether to stack states or return them individually. Defaults to `True`.
        with_keys
            Whether to return solution labelled by variables in form of a dictionary.
            Defaults to `False`.

        Returns
        -------
        Union[Dict[str, ndarray], Tuple[ndarray], ndarray]
            Steady state of system.
        """
        if (self.splicing_rate <= 0) or (self.degradation_rate <= 0):
            raise ValueError(
                "Both `splicing_rate` and `degradation_rate` need to be strictly positive."
            )
        else:
            unspliced = self.transcription_rate / self.splicing_rate
            spliced = self.transcription_rate / self.degradation_rate

        if with_keys:
            return {"u": unspliced, "s": spliced}
        elif not stacked:
            return unspliced, spliced
        else:
            return np.array([unspliced, spliced])


# TODO: Handle cases where denominator of ODE's solution == 0
class NucCytModel(DynamicsBase):
    """Splicing dynamics based on Nucleus-cytosol model.

    Arguments
    ---------
    transcription_rate
        Transcription rate.
    splicing_rate
        Translation rate.
    nuc_export_rate
        Nuclear export rate to cytoplasm.
    degradation_rate
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0, 0]`.

    Attributes
    ----------
    transcription_rate
        Transcription rate.
    splicing_rate
        Translation rate.
    nuc_export_rate
        Nuclear export rate to cytoplasm.
    degradation_rate
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0, 0]`.
    u0_nuc
        Initial abundance of unspliced RNA in nucleus.
    s0_nuc
        Initial abundance of spliced RNA in nucleus.
    s0_cyt
        Initial abundance of spliced RNA in cytoplasm.

    """

    def __init__(
        self,
        transcription_rate: float,
        splicing_rate: float,
        nuc_export_rate: float,
        degradation_rate: float,
        initial_state: Union[List, ndarray] = None,
    ):
        self.transcription_rate = transcription_rate
        self.splicing_rate = splicing_rate
        self.nuc_export_rate = nuc_export_rate
        self.degradation_rate = degradation_rate

        if initial_state is None:
            self.initial_state = [0, 0, 0]
        else:
            self.initial_state = initial_state

    @property
    def initial_state(self):
        """Function to get initial values as list [u0_nuc, s0_nuc, s0_cyt]."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        """Sets the initial values of undpliced and spliced abundances."""
        if isinstance(val, list) or (isinstance(val, ndarray) and (val.ndim == 1)):
            self.u0_nuc = val[0]
            self.s0_nuc = val[1]
            self.s0_cyt = val[2]
        else:
            self.u0_nuc = val[:, 0]
            self.s0_nuc = val[:, 1]
            self.s0_cyt = val[:, 2]
        self._initial_state = val

    def get_solution(
        self, t: ndarray, stacked: bool = True, with_keys: bool = False
    ) -> Union[Dict, ndarray]:
        """Calculate solution of dynamics.

        Arguments
        ---------
        t
            Time steps at which to evaluate solution.
        stacked
            Whether to stack states or return them individually. Defaults to `True`.
        with_keys
            Whether to return solution labelled by variables in form of a dictionary.
            Defaults to `False`.

        Returns
        -------
        Union[Dict, ndarray]
            Solution of system. If `with_keys=True`, the solution is returned in form of
            a dictionary with variables as keys. Otherwise, the solution is given as
            a `numpy.ndarray` of form `(n_steps, 2)`.
        """
        u_nuc = self.get_unspliced_nuc(
            t, self.u0_nuc, self.transcription_rate, self.splicing_rate
        )
        s_nuc = self.get_spliced_nuc(
            t,
            self.u0_nuc,
            self.s0_nuc,
            self.transcription_rate,
            self.splicing_rate,
            self.nuc_export_rate,
        )
        s_cyt = self.get_spliced_cyt(
            t,
            self.u0_nuc,
            self.s0_nuc,
            self.s0_cyt,
            self.transcription_rate,
            self.splicing_rate,
            self.nuc_export_rate,
            self.degradation_rate,
        )

        if with_keys:
            return {"u_nuc": u_nuc, "s_nuc": s_nuc, "s_cyt": s_cyt}
        elif not stacked:
            return u_nuc, s_nuc, s_cyt
        else:
            if isinstance(t, np.ndarray) and t.ndim == 2:
                return np.stack([u_nuc, s_nuc, s_cyt], axis=2)
            else:
                return np.column_stack([u_nuc, s_nuc, s_cyt])

    def get_unspliced_nuc(self, tau, u0_nuc, transcription_rate, splicing_rate):
        """Function returns unspliced RNA abundance in nucleus with the given solution to the ODE.

        Parameters
        ----------
        tau
            t_ig-t_0
        u0_nuc
            initial condition of u_nuc
        transcription_rate
            Transcription rate
        splicing_rate
            Splicing rate
        """
        expu = np.exp(-splicing_rate * tau)

        return u0_nuc * expu + transcription_rate / splicing_rate * (1 - expu)

    def get_spliced_nuc(
        self, tau, u0_nuc, s0_nuc, transcription_rate, splicing_rate, nuc_export_rate
    ):
        """Function returns spliced RNA abundance in nucleus with the given solution to the ODE.

        Parameters
        ----------
        tau
            t_ig-t_0
        u0_nuc
            initial condition of u_nuc
        s0_nuc
            initial condition of s_nuc
        transcription_rate
            Transcription rate
        splicing_rate
            Splicing rate
        nuc_export_rate
            Nuclear export rate
        """
        c = (transcription_rate - u0_nuc * splicing_rate) / (
            nuc_export_rate - splicing_rate
        )
        expu, exps = np.exp(-splicing_rate * tau), np.exp(-nuc_export_rate * tau)

        return (
            s0_nuc * exps
            + transcription_rate / nuc_export_rate * (1 - exps)
            + c * (exps - expu)
        )

    def get_spliced_cyt(
        self,
        tau,
        u0_nuc,
        s0_nuc,
        s0_cyt,
        transcription_rate,
        splicing_rate,
        nuc_export_rate,
        degradation_rate,
    ):
        """Function returns spliced RNA abundance in cytoplasm with the given solution to the ODE.

        Parameters
        ----------
        tau
            t_ig-t_0
        u0_nuc
            initial condition of u_nuc
        s0_nuc
            initial condition of s_nuc
        s0_cyt
            initial condition of s_cyt
        transcription_rate
            Transcription rate
        splicing_rate
            Splicing rate
        nuc_export_rate
            Nuclear export rate
        degradation_rate
            Degradation rate
        """
        spliced_cyt = (
            nuc_export_rate
            * splicing_rate
            * (
                (
                    transcription_rate
                    / splicing_rate
                    * (1 - np.exp(-splicing_rate * tau))
                    + u0_nuc * np.exp(-splicing_rate * tau)
                )
                / (
                    (nuc_export_rate - splicing_rate)
                    * (degradation_rate - splicing_rate)
                )
                - (
                    transcription_rate
                    / nuc_export_rate
                    * (1 - np.exp(-nuc_export_rate * tau))
                    + u0_nuc * np.exp(-nuc_export_rate * tau)
                )
                / (
                    (nuc_export_rate - splicing_rate)
                    * (degradation_rate - nuc_export_rate)
                )
                + (
                    transcription_rate
                    / degradation_rate
                    * (1 - np.exp(-degradation_rate * tau))
                    + u0_nuc * np.exp(-degradation_rate * tau)
                )
                / (
                    (degradation_rate - splicing_rate)
                    * (degradation_rate - nuc_export_rate)
                )
            )
            + (
                nuc_export_rate
                / (degradation_rate - nuc_export_rate)
                * (np.exp(-nuc_export_rate * tau) - np.exp(-degradation_rate * tau))
                * s0_nuc
            )
            + (np.exp(-degradation_rate * tau) * s0_cyt)
        )

        return spliced_cyt

    def get_steady_states(
        self, stacked: bool = True, with_keys: bool = False
    ) -> Union[Dict[str, ndarray], Tuple[ndarray], ndarray]:
        """Return steady state of system.

        Arguments
        ---------
        stacked
            Whether to stack states or return them individually. Defaults to `True`.
        with_keys
            Whether to return solution labelled by variables in form of a dictionary.
            Defaults to `False`.

        Returns
        -------
        Union[Dict[str, ndarray], Tuple[ndarray], ndarray]
            Steady state of system.
        """
        if (
            (self.splicing_rate <= 0)
            or (self.nuc_export_rate <= 0)
            or (self.degradation_rate <= 0)
        ):
            raise ValueError(
                "`splicing_rate`, `nuc_export_rate` and `degradation_rate` need to be strictly positive."
            )
        else:
            u_nuc = self.transcription_rate / self.splicing_rate
            s_nuc = self.transcription_rate / self.nuc_export_rate
            s_cyt = (
                self.nuc_export_rate
                * self.splicing_rate
                * (
                    u_nuc
                    / (
                        (self.nuc_export_rate - self.splicing_rate)
                        * (self.degradation_rate - self.splicing_rate)
                    )
                    - s_nuc
                    / (
                        (self.nuc_export_rate - self.splicing_rate)
                        * (self.degradation_rate - self.nuc_export_rate)
                    )
                    + (self.transcription_rate / self.degradation_rate)
                    / (
                        (self.degradation_rate - self.splicing_rate)
                        * (self.degradation_rate - self.nuc_export_rate)
                    )
                )
            )

        if with_keys:
            return {"u_nuc": u_nuc, "s_nuc": s_nuc, "s_cyt": s_cyt}
        elif not stacked:
            return u_nuc, s_nuc, s_cyt
        else:
            return np.array([u_nuc, s_nuc, s_cyt])
