"""
Cross-match DR9 North vs South surveys with separation < 1 arcsecond.

Memory strategy: NEVER load the full catalog.
  1. Parse RA/Dec bounds from sweep filenames.
  2. Find (north_file, south_file) pairs whose sky footprints overlap
     (with a small buffer for border objects).
  3. Each worker loads only two files, cross-matches them, writes a
     per-pair result FITS.
  4. A final pass concatenates all per-pair results + deduplicates.

Usage:
    python crossmatch_surveys.py --output matched.fits --workers 32
    python crossmatch_surveys.py --output matched.fits --workers 32 --tmpdir /scratch/xmatch_tmp
"""

import os
import re
import argparse
import logging
import tempfile
from itertools import product
from multiprocessing import Pool

import numpy as np
import fitsio
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SEP_LIMIT_ARCSEC = 1.0

COLS1 = [
    'BRICKID', 'OBJID', 'BRICKNAME', 'RA', 'DEC',
    'SHAPE_E1', 'SHAPE_E2', 'SHAPE_R',
    'FLUX_G', 'FLUX_R', 'FLUX_Z',
    'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
    'SERSIC', 'TYPE',
    'NOBS_G', 'NOBS_R', 'NOBS_Z',
    'MASKBITS',
    'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z',
    'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
    'FRACIN_G', 'FRACIN_R', 'FRACIN_Z',
    'GAIA_PHOT_G_MEAN_MAG',
    'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z',
    'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4',
    'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
    'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4',
    'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
    'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z',
    'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3',
    'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
    'EBV',
]
COLS2 = ['Z_PHOT_MEDIAN', 'Z_PHOT_L95']


# ---------------------------------------------------------------------------
# Filename → sky-bounds parsing
# sweep-{ra_min}p/m{dec_min}-{ra_max}p/m{dec_max}.fits
# e.g. sweep-350p030-360p035.fits  -> RA [350,360], Dec [3.0, 3.5]
#      sweep-000m100-010m050.fits  -> RA [0,10],   Dec [-10.0, -5.0]
# ---------------------------------------------------------------------------
_SWEEP_RE = re.compile(
    r"sweep-(\d{3})(p|m)(\d{3})-(\d{3})(p|m)(\d{3})(?:-pz)?\.fits$"
)

def parse_bounds(fname: str):
    """Return (ra_min, ra_max, dec_min, dec_max) or None."""
    m = _SWEEP_RE.search(os.path.basename(fname))
    if not m:
        return None
    ra1, s1, dec1, ra2, s2, dec2 = m.groups()
    sign1 = 1 if s1 == 'p' else -1
    sign2 = 1 if s2 == 'p' else -1
    # DR9 sweep filenames encode Dec×10 as an integer, e.g. "030" = 3.0 deg
    return (float(ra1), float(ra2),
            sign1 * float(dec1) / 10,
            sign2 * float(dec2) / 10)


def boxes_overlap(b1, b2, buf_deg: float = 0.1) -> bool:
    """True if two (ra_min,ra_max,dec_min,dec_max) boxes overlap (+buffer)."""
    ra1lo, ra1hi, de1lo, de1hi = b1
    ra2lo, ra2hi, de2lo, de2hi = b2

    # Dec overlap (simple interval test)
    if de1hi + buf_deg < de2lo or de2hi + buf_deg < de1lo:
        return False

    # RA overlap with wrap handling
    def _ra_overlap(alo, ahi, blo, bhi, buf):
        if ahi - alo >= 360 or bhi - blo >= 360:
            return True
        shift = alo
        blo_s = (blo - shift) % 360
        bhi_s = (bhi - shift) % 360
        width = (ahi - shift) % 360 or 360
        if bhi_s < blo_s:
            bhi_s += 360
        return blo_s <= width + buf and bhi_s >= -buf

    return _ra_overlap(ra1lo, ra1hi, ra2lo, ra2hi, buf_deg)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def get_file_info(directory: str):
    """Return list of (fits_path, pz_path_or_None, bounds) for a hemisphere."""
    d1 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0/"
    d2 = f"/storage/shadab/data/legacy_survey/dr9/{directory}/sweep/9.0-photo-z/"
    records = []
    for fname in sorted(os.listdir(d1)):
        if not fname.endswith(".fits") or "-pz" in fname:
            continue
        bounds = parse_bounds(fname)
        if bounds is None:
            log.warning("Cannot parse bounds from '%s' — skipping", fname)
            continue
        pz_full = os.path.join(d2, fname.replace(".fits", "-pz.fits"))
        records.append((
            os.path.join(d1, fname),
            pz_full if os.path.exists(pz_full) else None,
            bounds,
        ))
    return records


# ---------------------------------------------------------------------------
# Per-pair worker (runs in its own process — max memory = 2 sweep files)
# ---------------------------------------------------------------------------
def process_pair(args):
    n_fits, n_pz, s_fits, s_pz, tmp_dir, sep_limit = args

    tag = (os.path.basename(n_fits).replace(".fits", "")
           + "__VS__"
           + os.path.basename(s_fits).replace(".fits", ""))
    out_path = os.path.join(tmp_dir, tag + "_matched.fits")

    # Resume support: skip if already processed
    if os.path.exists(out_path):
        log.info("SKIP (exists) %s", tag)
        return out_path

    try:
        # ---- load north tile ------------------------------------------------
        tn = Table(fitsio.FITS(n_fits)[1].read(columns=COLS1))
        if n_pz:
            tn_pz = Table(fitsio.FITS(n_pz)[1].read(columns=COLS2))
            for c in COLS2:
                tn[c] = tn_pz[c]

        # ---- load south tile ------------------------------------------------
        ts = Table(fitsio.FITS(s_fits)[1].read(columns=COLS1))
        if s_pz:
            ts_pz = Table(fitsio.FITS(s_pz)[1].read(columns=COLS2))
            for c in COLS2:
                ts[c] = ts_pz[c]

        if len(tn) == 0 or len(ts) == 0:
            return None

        # ---- cross-match (KD-tree via astropy) ------------------------------
        cn = SkyCoord(ra=tn['RA'] * u.deg, dec=tn['DEC'] * u.deg)
        cs = SkyCoord(ra=ts['RA'] * u.deg, dec=ts['DEC'] * u.deg)

        idx, sep2d, _ = cn.match_to_catalog_sky(cs)
        mask = sep2d < sep_limit * u.arcsec

        if mask.sum() == 0:
            return None

        matched_n = tn[mask]
        matched_s = ts[idx[mask]]
        matched_n['SEP_ARCSEC'] = sep2d[mask].to(u.arcsec).value

        for col in matched_s.colnames:
            matched_n[col + '_S'] = matched_s[col]

        matched_n.write(out_path, format='fits', overwrite=True)
        log.info("Matched %6d rows  %s", mask.sum(), tag)
        return out_path

    except Exception as exc:
        log.error("FAILED %s : %s", tag, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Tile-by-tile cross-match DR9 North vs South (memory-safe).")
    parser.add_argument("--output", default="matched_north_south.fits")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sep", type=float, default=SEP_LIMIT_ARCSEC,
                        help="Match radius in arcseconds")
    parser.add_argument("--tmpdir", default=None,
                        help="Stable dir for per-pair files; enables resume.")
    args = parser.parse_args()

    tmp_dir = args.tmpdir or tempfile.mkdtemp(prefix="crossmatch_tmp_")
    os.makedirs(tmp_dir, exist_ok=True)
    log.info("Intermediate files → %s", tmp_dir)

    log.info("Discovering files …")
    north_files = get_file_info("north")
    south_files = get_file_info("south")
    log.info("North: %d tiles   South: %d tiles", len(north_files), len(south_files))

    # Build overlapping (north_tile, south_tile) pairs
    pairs = [
        (nf, np_, sf, sp, tmp_dir, args.sep)
        for (nf, np_, nb), (sf, sp, sb) in product(north_files, south_files)
        if boxes_overlap(nb, sb)
    ]
    log.info("Overlapping tile pairs: %d", len(pairs))

    # Each worker holds at most 2 sweep files in RAM at a time
    with Pool(processes=args.workers) as pool:
        result_paths = pool.map(process_pair, pairs)

    valid = [p for p in result_paths if p and os.path.exists(p)]
    log.info("Collecting %d result files …", len(valid))

    if not valid:
        log.warning("No matches found!")
        return

    log.info("Stacking results …")
    final = vstack([Table.read(p) for p in valid])
    log.info("Total rows before dedup : %d", len(final))

    # Deduplicate: a north object might appear in several overlapping tile pairs
    keys = np.array(list(zip(final['BRICKID'].data, final['OBJID'].data)))
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    final = final[uniq_idx]
    log.info("Total rows after  dedup : %d", len(final))

    final.write(args.output, format='fits', overwrite=True)
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()