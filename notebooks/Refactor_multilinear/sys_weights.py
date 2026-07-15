"""
sys_weights.py — imaging-systematics weight estimation via pixel-level WLS regression.

Fixes relative to the original multilinear_regression.ipynb (see chat summary):
  * pixels with ngal = 0 are KEPT in the fit (dropping them biases coefficients
    toward zero exactly where contamination is strongest)
  * variance model uses EXPECTED counts, sigma^2(delta) = 1/(nbar * nran),
    not observed sqrt(ngal) (observed-count weighting correlates noise with
    signal and is undefined at ngal = 0)
  * intercept included (centered X without an intercept lets correlated
    features absorb the offset)
  * no dense NxN diagonal weight matrix (broadcasting instead — the old
    np.diag(1/sig**2) built a multi-GB matrix for nothing)
  * optional coverage cut on random counts (boundary pixels with tiny
    fractional coverage have delta noise the Poisson model underestimates)
  * optional ridge regularisation for the strongly collinear PSFDEPTH set
  * per-sample refitting supported: pass any (ra, dec) subset — this is what
    enables per-property-bin weights in the binned w(theta) pipeline

Usage (global):
    sw = SysWeights(part='south', mask_path=..., sys_dir=..., rand_counts_path=...)
    fit = sw.fit(ra, dec)                  # full-sample weights
    wmap = fit['weight_map']               # healpix map, 1 outside fit pixels

Usage (per bin):
    fit_b = sw.fit(ra[mask_bin], dec[mask_bin])   # same footprint, bin's own coeffs
"""

import numpy as np
import healpy as hp


DEFAULT_SYSTEMATICS = ['H_Alpha', 'Saggitarius_Density', 'Stellar_Density_AllWISE',
                       'CSFD', 'PSFDEPTH_Z', 'PSFDEPTH_G', 'PSFDEPTH_R',
                       'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_W1']

# per-systematic upper percentile for 'partial_percentile' outlier cuts
_UPPER_PCT = {'CSFD': 0.95, 'Saggitarius_Density': 0.95,
              'Stellar_Density_AllWISE': 0.95, 'H_Alpha': 0.95}
_UPPER_PCT_DEFAULT = 0.99
_LOWER_PCT = 0.01


def points_to_map(ra_deg, dec_deg, nside, weights=None):
    ipix = hp.ang2pix(nside, np.radians(90.0 - np.asarray(dec_deg)),
                      np.radians(np.asarray(ra_deg)))
    return np.bincount(ipix, weights=weights,
                       minlength=hp.nside2npix(nside)).astype(np.float64)


class SysWeights:
    def __init__(self, part, mask_path, sys_dir, rand_counts_path,
                 systematics=DEFAULT_SYSTEMATICS, nside=64,
                 coverage_min=0.2, outlier_kind='partial_percentile',
                 outlier_val=5.0, quadratic=None,
                 upper_pct=None, lower_pct=_LOWER_PCT):
        """outlier_kind: 'partial_percentile' (asymmetric, per-systematic upper
        cuts), 'equal' (symmetric, outlier_val percent removed in total, split
        between the two tails), or 'none' (full sample).

        quadratic: list of systematic names that also get a squared
        (standardised) term in the design matrix — for maps whose density
        response is visibly nonlinear (e.g. Stellar_Density_AllWISE,
        PSFDEPTH_W1 for a WISE-selected sample). Modelling the curvature lets
        the outlier cuts on those maps be loosened, recovering sky area.

        upper_pct / lower_pct: override the per-map percentile cuts, e.g.
        upper_pct={'CSFD': 0.98} with a default of 0.995 elsewhere by passing
        upper_pct={'_default': 0.995, 'CSFD': 0.98}. With 11 maps the cuts
        COMPOUND (~0.94^11 ≈ 0.5 of the sky), so per-map values should be as
        loose as the modelling allows.

        part: 'north' | 'south' — used only for filename construction here;
        the caller decides which galaxies to pass in (keep the SAME dec split
        as the clustering pipeline — the original code used 32.3 here and
        32.375 in w(theta), leaving a mis-weighted strip)."""
        self.part, self.nside = part, nside
        self.outlier_kind, self.outlier_val = outlier_kind, outlier_val
        self.quadratic = list(quadratic) if quadratic else []
        _up = dict(_UPPER_PCT)
        if upper_pct:
            default = upper_pct.pop('_default', _UPPER_PCT_DEFAULT)
            _up.update(upper_pct)
        else:
            default = _UPPER_PCT_DEFAULT
        self._upper_pct, self._upper_default, self._lower_pct = _up, default, lower_pct
        self.npix = hp.nside2npix(nside)
        self.systematics = list(systematics)
        self.mask = hp.read_map(mask_path) > 0
        self.nran_map = hp.read_map(rand_counts_path).astype(np.float64)

        self.sys_maps = {}
        for s in self.systematics:
            m = hp.read_map(f'{sys_dir}/{part}_{s}_nside_{nside}.fits')
            m = np.where(np.isclose(m, hp.UNSEEN) | ~np.isfinite(m), np.nan, m)
            self.sys_maps[s] = m

        # ── footprint pixels usable for fitting (galaxy-independent) ────────
        cov = self.nran_map / np.nanmax(self.nran_map[self.mask])
        good = self.mask & (cov > coverage_min)
        for s in self.systematics:
            good &= np.isfinite(self.sys_maps[s])
        # the ANALYSIS footprint: coverage+finite only, BEFORE outlier cuts —
        # this is the sky an extrapolated weight map can cover
        self.analysis_pixels = np.where(good)[0]

        # outlier cuts on systematic values. The excluded pixels are dropped
        # from the FIT; whether they are also dropped from the analysis is the
        # caller's choice via fit(extrapolate=...): extrapolate=False exports
        # only the fit footprint (mask them in clustering too), while
        # extrapolate=True evaluates the fitted model on the excluded pixels
        # so their galaxies keep a (model-extrapolated) weight.
        self.limits = {}
        if outlier_kind in ('partial_percentile', 'equal'):
            base = good.copy()          # limits from one common base footprint
            for s in self.systematics:
                v = self.sys_maps[s][base]
                if outlier_kind == 'partial_percentile':
                    up = self._upper_pct.get(s, self._upper_default)
                    lo, hi = np.quantile(v, [self._lower_pct, up])
                else:
                    half = outlier_val / 100.0 / 2.0
                    lo, hi = np.quantile(v, [half, 1.0 - half])
                self.limits[s] = (lo, hi)
                good &= (self.sys_maps[s] >= lo) & (self.sys_maps[s] <= hi)
        elif outlier_kind != 'none':
            raise ValueError(f"unknown outlier_kind '{outlier_kind}'")
        self.fit_pixels = np.where(good)[0]

        # standardisation stats from the FIT sample (extrapolation reuses them)
        X = np.column_stack([self.sys_maps[s][self.fit_pixels]
                             for s in self.systematics])
        self._Xmean, self._Xstd = X.mean(axis=0), X.std(axis=0)
        Xstd = (X - self._Xmean) / self._Xstd
        self._qstats = []
        for s in self.quadratic:
            j = self.systematics.index(s)
            q = Xstd[:, j]**2
            self._qstats.append((q.mean(), q.std()))
        self.feature_names = (['intercept'] + list(self.systematics)
                              + [f'{s}^2' for s in self.quadratic])
        self.X = self.design(self.fit_pixels)   # fit design matrix
        self.nran = self.nran_map[self.fit_pixels]

    def design(self, pixels):
        """Design matrix for any pixel set, standardised with the FIT-sample
        statistics — evaluating it on non-fit pixels is the extrapolation."""
        X = np.column_stack([self.sys_maps[s][pixels] for s in self.systematics])
        Xstd = (X - self._Xmean) / self._Xstd
        cols = [np.ones(len(pixels)), Xstd]
        for s, (qm, qs) in zip(self.quadratic, self._qstats):
            j = self.systematics.index(s)
            q = Xstd[:, j]**2
            cols.append(((q - qm) / qs)[:, None])
        return np.column_stack(cols)

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, ra, dec, ridge=0.0, clip=(0.5, 2.0), extrapolate=False):
        """WLS fit of delta(pixel) against systematics for the given galaxies.

        Works for the full sample or any subset (a property bin).

        extrapolate=False: weights are defined only on the fit footprint
        (outlier pixels must then be masked in the clustering, data AND
        randoms — no sky kept, no extrapolation risk).
        extrapolate=True: the fitted model is EVALUATED on the outlier pixels
        too, so the most contaminated sky stays in the analysis with a
        model-extrapolated weight. The coefficients are still trained only on
        the cut footprint (leverage points stay out of the fit). The `clip`
        bounds cap runaway extrapolated corrections; 'extrap_clipped_frac'
        reports how often that cap engages — if it is large, the model does
        not extend to those pixels and cutting (or a quadratic term) is the
        honest choice. ALWAYS validate with null_test(..., pixels='all').

        Returns coefficients (standardised features), errors, chi2 before /
        after, the weight map, and two masks: fit_pixel_mask (where the model
        was trained) and weight_pixel_mask (where weights are valid for
        analysis — equals the analysis footprint when extrapolating).
        """
        ngal_map = points_to_map(ra, dec, self.nside)
        ngal = ngal_map[self.fit_pixels]
        nbar = ngal.sum() / self.nran.sum()
        mu = nbar * self.nran                       # expected counts per pixel
        y = ngal / mu - 1.0                         # overdensity (ngal=0 kept!)
        var = 1.0 / mu                              # Var(delta) = mu/mu^2
        w = 1.0 / var

        Xw = self.X * w[:, None]
        A = Xw.T @ self.X
        if ridge > 0:
            pen = np.ones(self.X.shape[1]); pen[0] = 0.0   # all but intercept
            A = A + ridge * np.diag(pen)
        b = Xw.T @ y
        coeffs = np.linalg.solve(A, b)
        cov_c = np.linalg.inv(A)
        errs = np.sqrt(np.diag(cov_c))

        pred = self.X @ coeffs                      # predicted contamination
        chi2_before = float(np.sum(y**2 * w))
        chi2_after = float(np.sum((y - pred)**2 * w))
        ndof = len(y) - self.X.shape[1]

        # pixels the weights will cover
        if extrapolate:
            use_pixels = self.analysis_pixels
            pred_use = self.design(use_pixels) @ coeffs
        else:
            use_pixels = self.fit_pixels
            pred_use = pred
        raw_w = 1.0 / (1.0 + pred_use)
        wpix = np.clip(raw_w, *clip) if clip is not None else raw_w
        extrap_clipped_frac = 0.0
        if extrapolate and clip is not None:
            in_fit = np.isin(use_pixels, self.fit_pixels)
            outl = ~in_fit
            if outl.any():
                extrap_clipped_frac = float(np.mean(
                    (raw_w[outl] <= clip[0]) | (raw_w[outl] >= clip[1])))
        mu_use = nbar * self.nran_map[use_pixels]
        wpix = wpix / np.average(wpix, weights=mu_use)   # mean 1 over exp. counts

        wmap = np.ones(self.npix)
        wmap[use_pixels] = wpix
        fitmask = np.zeros(self.npix, dtype=bool)
        fitmask[self.fit_pixels] = True
        usemask = np.zeros(self.npix, dtype=bool)
        usemask[use_pixels] = True

        return dict(coeffs=coeffs[1:], intercept=coeffs[0], errors=errs[1:],
                    systematics=self.systematics,
                    feature_names=self.feature_names[1:],
                    chi2_before=chi2_before, chi2_after=chi2_after, ndof=ndof,
                    weight_map=wmap, fit_pixel_mask=fitmask,
                    weight_pixel_mask=usemask, extrapolated=bool(extrapolate),
                    extrap_clipped_frac=extrap_clipped_frac, nbar=nbar)

    # ─────────────────────────────────────────────────────────────────────────
    def galaxy_weights(self, fit, ra, dec):
        """Per-galaxy weights from a fit's map. Galaxies outside the weight
        footprint get NaN so the caller must decide: exclude them (and mask
        the same pixels in the randoms) or set 1 (leaves them uncorrected).
        With extrapolate=True the weight footprint covers the outlier pixels,
        so NaNs only occur off the coverage/finite footprint."""
        pix = hp.ang2pix(self.nside, np.radians(90.0 - np.asarray(dec)),
                         np.radians(np.asarray(ra)))
        valid = fit.get('weight_pixel_mask', fit['fit_pixel_mask'])
        w = fit['weight_map'][pix].copy()
        w[~valid[pix]] = np.nan
        return w

    def null_test(self, fit, ra, dec, nbins=20, nside_jk=8, pixels='fit'):
        """Overdensity vs each systematic, before/after weighting, with chi2
        against flat. Run this per property bin with GLOBAL weights first —
        it tells you which bins actually need their own fit.

        pixels: 'fit' evaluates on the fit footprint; 'all' on the full
        analysis footprint INCLUDING outlier pixels — mandatory validation
        when extrapolate=True (the tails of each panel then show whether the
        extrapolated weights flatten the relation beyond the cuts, which the
        fit never saw).

        Errors are SKY-JACKKNIFE (delete-one nside_jk super-pixel region), not
        Poisson: at nside=64 each pixel carries real clustering variance, and
        systematic values are spatially coherent, so Poisson-only errors make
        chi2/bin land at 10-100 even for a clean, corrected sample. Jackknife
        errors include the clustering contribution and make the chi2 values
        interpretable (~1/bin expected for a flat relation)."""
        px = self.analysis_pixels if pixels == 'all' else self.fit_pixels
        Xd = self.design(px) if pixels == 'all' else self.X
        nran = self.nran_map[px]
        ngal = points_to_map(ra, dec, self.nside)[px]
        pix = hp.ang2pix(self.nside, np.radians(90.0 - np.asarray(dec)),
                         np.radians(np.asarray(ra)))
        wg = fit['weight_map'][pix]
        ngal_w = points_to_map(ra, dec, self.nside, weights=wg)[px]

        # jackknife region id per pixel = parent low-res healpix pixel
        th, ph = hp.pix2ang(self.nside, px)
        jk_all = hp.ang2pix(nside_jk, th, ph)
        regions, jk = np.unique(jk_all, return_inverse=True)
        nreg = len(regions)

        out = {}
        for k, s in enumerate(self.systematics):
            v = Xd[:, k + 1]                        # standardised linear feature
            edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
            idx = np.clip(np.digitize(v, edges[1:-1]), 0, nbins - 1)
            code = idx * nreg + jk                  # (bin, region) combined index
            res = {}
            for tag, ng in (('before', ngal), ('after', ngal_w)):
                nbar = ng.sum() / nran.sum()
                num_br = np.bincount(code, weights=ng,
                                     minlength=nbins*nreg).reshape(nbins, nreg)
                den_br = np.bincount(code, weights=nbar * nran,
                                     minlength=nbins*nreg).reshape(nbins, nreg)
                num, den = num_br.sum(axis=1), den_br.sum(axis=1)
                d = num / den - 1.0
                with np.errstate(divide='ignore', invalid='ignore'):
                    d_jk = (num[:, None] - num_br) / (den[:, None] - den_br) - 1.0
                d_jk = np.where(np.isfinite(d_jk), d_jk, d[:, None])
                dbar = d_jk.mean(axis=1)
                sig = np.sqrt((nreg - 1) / nreg
                              * np.sum((d_jk - dbar[:, None])**2, axis=1))
                sig = np.where(sig > 0, sig, np.inf)
                res[tag] = (d, sig, float(np.sum((d / sig)**2)))
            centers = 0.5 * (edges[:-1] + edges[1:])
            out[s] = dict(centers=centers, before=res['before'], after=res['after'])
        return out