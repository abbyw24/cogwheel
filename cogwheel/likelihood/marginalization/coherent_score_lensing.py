from functools import wraps
from dataclasses import dataclass
import numpy as np

from .coherent_score_hm import CoherentScoreHM, _flip_psi
from .base import MarginalizationInfo
from cogwheel import utils


def _flip_psi_lensed(psi, d_h, flip_psi, lensed):
    """
    Note input `d_h` is complex, but only a scalar d_h is returned, depending on
    whether the signal is lensed.
    """
    # handle psi
    if flip_psi:
        psi = (psi + np.pi / 2) % np.pi

    # pick component of d_h based on lensing
    d_h = np.imag(d_h) if lensed else np.real(d_h)

    # apply sign flip if needed
    if flip_psi:
        d_h = -d_h

    return psi, d_h


_flip_psi_lensed = utils.handle_scalars(np.vectorize(_flip_psi_lensed,
                                              otypes=[float, float]))


class CoherentScoreLensing(CoherentScoreHM):

    """
    Class that, given a matched-filtering timeseries, computes the
    likelihood marginalized over extrinsic parameters
    (:py:meth:`get_marginalization_info`).

    Extrinsic parameters samples can be generated as well
    (:py:meth:`gen_samples_from_marg_info`).

    Works for quasi-circular waveforms with generic spins and higher
    modes.

    Inherits from parent CoherentScoreHM.

    """

    def _get_dh_hh_qo(self, sky_inds, q_inds, t_first_det, times,
                        dh_mptd, hh_mppd): 

        dh_qm, hh_qm = self._get_dh_hh_qm(sky_inds, q_inds, t_first_det, times,
                                            dh_mptd, hh_mppd)

        dh_qo = dh_qm @ self._dh_phasor
        hh_qo = utils.real_matmul(hh_qm, self._hh_phasor)

        return dh_qo, hh_qo

    def _get_lnnumerators_important_flippsi(self, dh_qo, hh_qo, sky_prior):
        """
        Parameters
        ----------
        dh_qo : (n_physical, n_phi) float array
            ⟨d|h⟩ real inner product between data and waveform at
            ``self.lookup_table.REFERENCE_DISTANCE``.

        hh_qo : (n_physical, n_phi) float array
            ⟨h|h⟩ real inner product of a waveform at
            ``self.lookup_table.REFERENCE_DISTANCE`` with itself.

        sky_prior : (n_physical,) float array
            Prior weights of the QMC sequence.

        Return
        ------
        ln_numerators : float array of length n_important
            Natural log of the weights of the QMC samples, including the
            likelihood and prior but excluding the importance sampling
            weights.

        important : (array of ints, array of ints) of lengths n_important
            The first array contains indices between 0 and n_physical-1
            corresponding to (physical) QMC samples.
            The second array contains indices between 0 and n_phi-1
            corresponding to orbital phases.
            They correspond to samples with sufficiently high maximum
            likelihood over distance to be included in the integral.

        flip_psi : bool array of length n_important
            Whether to add pi/2 to psi (which inverts the sign of ⟨d|h⟩
            and preserves ⟨h|h⟩).

        lensed : bool array of length n_important
            Whether to multiply the waveform by a factor of i (⟨d|h⟩-> i⟨d|h⟩).
        """

        # dh_qo.shape == (n_physical, n_phi)
        dh_slqo = np.array([
            [np.real(dh_qo), np.imag(dh_qo)],
            [-np.real(dh_qo), -np.imag(dh_qo)]
        ])
        # dh_slqo.shape == (2, 2, n_physical, n_phi)
        # flip_psi = np.array([
        #     [False, False],
        #     [True, True]
        # ])
        # lensed = np.array([
        #     [False, True],
        #     [False, True]
        # ])
        flip_psi = np.full(dh_slqo.shape, False)
        flip_psi[1] = True
        lensed = np.full(dh_slqo.shape, False)
        lensed[:, 1] = True

        max_over_distance_lnl = np.zeros_like(dh_slqo)
        idx_positive = np.where(dh_slqo > 0)
        max_over_distance_lnl[idx_positive] = (
            0.5 * dh_slqo[idx_positive]**2 / hh_qo[idx_positive[2], idx_positive[3]])
        # flip_psi = np.signbit(dh_qo)  # qo
        # flip_lensing = [np.abs(np.imag(dh_qo_)) > np.abs(np.real(dh_qo_)) for dh_qo_ in dh_qo]
        # max_over_distance_lnl = 0.5 * dh_qo**2 / hh_qo  # qo
        threshold = np.max(max_over_distance_lnl) - self.DLNL_THRESHOLD
        important = np.where(max_over_distance_lnl > threshold)
        important_qo = (important[2], important[3])

        ln_numerators = (
            self.lookup_table.lnlike_marginalized(dh_slqo[important],
                                                  hh_qo[important_qo])
            + np.log(sky_prior)[important[0]]
            - np.log(self._nphi))  # i

        return ln_numerators, important_qo, flip_psi[important], lensed[important]

    def _get_marginalization_info_chunk(self, d_h_timeseries, h_h,
                                        times, t_arrival_prob, i_chunk):
        """
        Evaluate inner products (d|h) and (h|h) at integration points
        over a chunk of a QMC sequence of extrinsic parameters, given
        timeseries of (d|h) and value of (h|h) by mode `m`, polarization
        `p` and detector `d`.

        Parameters
        ----------
        d_h_timeseries : (n_m, 2, n_t, n_d) complex array
            Timeseries of complex (d|h), inner product of data against a
            waveform at reference distance and phase.
            Decomposed by mode, polarization, time, detector.

        h_h : (n_mm, 2, 2, n_d) complex array
            Complex (h|h) inner product of a waveform with itself,
            decomposed by mode, polarization and detector.

        times : (n_t,) float array
            Timestamps of the timeseries (s).

        t_arrival_prob : (n_d, n_t) float array
            Proposal probability of time of arrival at each detector,
            normalized to sum to 1 along the time axis.

        i_chunk : int
            Index to ``._qmc_ind_chunks``.

        Returns
        -------
        Instance of ``MarginalizationInfoHM`` with several fields, see
        its documentation.
        """
        if d_h_timeseries.shape[0] != self.m_arr.size:
            raise ValueError('Incorrect number of harmonic modes.')

        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(
                tdet_inds[1:] - tdet_inds[0])  # q, q, q

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]

        if not any(physical_mask):
            return MarginalizationInfoLensing(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool),
                lensed=np.array([], bool)
                )

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, q_inds, t_first_det,
                                          times, d_h_timeseries, h_h)  # qo, qo

        ln_numerators, important, flip_psi, lensed \
            = self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]

        return MarginalizationInfoLensing(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
            lensed=lensed
            )

    def gen_samples_from_marg_info(self, marg_info, num=()):
        """
        Generate requested number of extrinsic parameter samples.

        Parameters
        ----------
        marg_info : MarginalizationInfoHM or None
            Normally, output of ``.get_marginalization_info``.
            If ``None``, assume that the sampled parameters were unphysical
            and return samples full of nans.

        num : int, optional
            Number of samples to generate, ``None`` makes a single sample.

        Returns
        -------
        samples : dict
            Values are scalar if `num` is ``None``, else numpy arrays.
            If ``marg_info`` correspond to an unphysical sample (i.e.,
            a realization of matched-filtering timeseries in the
            detectors incompatible with a real signal) the values will
            be NaN.
        """
        if marg_info is None or marg_info.q_inds.size == 0:
            # Order and dtype must match that of regular output
            out = dict.fromkeys(['d_luminosity', 'dec', 'lon', 'phi_ref',
                                 'psi', 't_geocenter', 'lnl_marginalized',
                                 'lnl', 'h_h', 'n_effective', 'n_qmc', 'p_lensed'],
                                np.full(num, np.nan)[()])
            if marg_info is None:
                out['n_qmc'] = np.full(num, 0)[()]
            else:
                out['lnl_marginalized'] = np.full(
                    num, marg_info.lnl_marginalized)[()]
                out['n_effective'] = np.full(num, marg_info.n_effective)[()]
                out['n_qmc'] = np.full(num, marg_info.n_qmc)[()]
            return out

        self._switch_qmc_sequence(marg_info.qmc_sequence_id)
        random_ids = self._rng.choice(len(marg_info.q_inds), size=num,
                                      p=marg_info.weights)[()]

        q_ids = marg_info.q_inds[random_ids]
        o_ids = marg_info.o_inds[random_ids]
        sky_ids = marg_info.sky_inds[random_ids]
        t_geocenter = (marg_info.t_first_det[random_ids]
                       - self.sky_dict.geocenter_delay_first_det[sky_ids])
        h_h = marg_info.h_h[random_ids]

        psi, d_h = _flip_psi_lensed(self._qmc_sequence['psi'][q_ids],
                             marg_info.d_h[random_ids],
                             marg_info.flip_psi[random_ids],
                             marg_info.lensed[random_ids])

        d_luminosity = self._sample_distance(d_h, h_h)
        distance_ratio = d_luminosity / self.lookup_table.REFERENCE_DISTANCE

        return {
            'd_luminosity': d_luminosity,
            'dec': self.sky_dict.sky_samples['lat'][sky_ids],
            'lon': self.sky_dict.sky_samples['lon'][sky_ids],
            'phi_ref': self._phi_ref[o_ids],
            'psi': psi,
            't_geocenter': t_geocenter,
            'lnl_marginalized': np.full(num, marg_info.lnl_marginalized)[()],
            'lnl': d_h / distance_ratio - h_h / distance_ratio**2 / 2,
            'h_h': h_h / distance_ratio**2,
            'n_effective': np.full(num, marg_info.n_effective)[()],
            'n_qmc': np.full(num, marg_info.n_qmc)[()],
            'p_lensed' : np.full(num, marg_info.p_lensed)[()]}

@dataclass
class MarginalizationInfoLensing(MarginalizationInfo):
    """
    Like `MarginalizationInfo` except:

    * it additionally contains ``.o_inds``
    * ``.d_h`` has dtype float, not complex.

    Attributes
    ----------
    o_inds : int array of length n_important
        Indices to the orbital phase.

    d_h : float array of length n_important
        Real inner product ⟨d|h⟩.

    flip_psi : int array of length n_important
        Whether to add pi/2 to psi.

    lensed : int array of length n_important
        Whether to multiply the waveform by i.
    """
    o_inds: np.ndarray
    flip_psi: np.ndarray
    lensed: np.ndarray

    def __post_init__(self):
        """Set derived attributes, including probability of Type II lensed."""

        super().__post_init__()

        self.p_lensed = np.sum(self.weights[self.lensed])

    @wraps(MarginalizationInfo.update)
    def update(self, other):
        self.o_inds = np.concatenate([self.o_inds, other.o_inds])
        self.flip_psi = np.concatenate([self.flip_psi, other.flip_psi])
        self.lensed = np.concatenate([self.lensed, other.lensed])
        super().update(other)

def bayes_factor_p_lensed(distribution, weights=None):
    avg_p_lensed = np.average(distribution, weights=weights)
    return avg_p_lensed / (1 - avg_p_lensed)

def get_bayes_factor_of_samples(samples, weights_col='weights'):
    """
    Calculate the Bayes factor for Type II lensing for a set of samples with `p_lensed`.
    """
    return bayes_factor_p_lensed(samples['p_lensed'], weights=samples[weights_col])

def bootstrap_bayes_factors(samples, n_resamples, weights_col='weights', n_to_resample=None):
    """
    Use bootstrap to return a distribution of Bayes factors for Type II lensing from a set of posterior samples.
    If `n_to_resample` is `None`, the size of each resample matches the number of original samples.
    """
    bayes_factors = []
    resample_size = n_to_resample if n_to_resample is not None else len(samples)

    for i in range(n_resamples):
        # resample from the p_lensed distribution:
        # shuffle the indices
        idx = np.random.choice(len(samples), size=len(samples), replace=True)
        resampled_p_lensed, reweights = samples['p_lensed'][idx], samples['weights'][idx]
            # remember we also have to shuffle the weights in the same way
        # get the bayes factor of the resampled distribution:
        bayes_factors.append(bayes_factor_p_lensed(resampled_p_lensed, weights=reweights))

    return np.array(bayes_factors)