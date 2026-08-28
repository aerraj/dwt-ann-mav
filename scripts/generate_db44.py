"""Generate minimum-phase Daubechies filters by high-precision spectral factorization.

P(y) = sum(binomial(N-1+k, k) y**k), y=(2-z-z**-1)/4.
Select the N-1 roots outside the unit circle, add N roots at -1,
normalize the coefficient sum to sqrt(2). Stored order is analysis dec_lo.
No double-precision polynomial roots are used. mpmath is a build-only dependency.
"""

import argparse
import json
from math import comb
from pathlib import Path

import mpmath as mp


def coefficients(order, precision=100):
    with mp.workdps(precision):
        p = [mp.mpf(comb(order - 1 + k, k)) for k in reversed(range(order))]
        roots = mp.polyroots(p, maxsteps=3000, extraprec=500, error=False)
        poly = [mp.mpf(1)]
        zroots = []
        for y in roots:
            z = 1 - 2 * y + 2 * mp.sqrt(y * (y - 1))
            if abs(z) < 1:
                z = 1 / z
            zroots.append(z)
        for root in zroots + [-mp.mpf(1)] * order:
            out = [mp.mpc(0)] * (len(poly) + 1)
            for i, value in enumerate(poly):
                out[i] += value
                out[i + 1] -= root * value
            poly = out
        norm = mp.sqrt(2) / sum(poly)
        result = [mp.re(c * norm) for c in poly]
        # Orthonormal even shifts, including unit energy.
        for shift in range(0, 2 * order, 2):
            dot = sum(result[i] * result[i + shift] for i in range(2 * order - shift))
            assert abs(dot - (1 if shift == 0 else 0)) < mp.mpf("1e-70")
        return [float(c) for c in result]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=44)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    low = coefficients(args.order)
    args.output.write_text(
        json.dumps(
            {
                "name": f"db{args.order}",
                "order": args.order,
                "precision_decimal_digits": 100,
                "method": "minimum-phase Daubechies spectral factorization",
                "dec_lo": low,
                "dec_hi": [(-1 if i % 2 == 0 else 1) * x for i, x in enumerate(reversed(low))],
            },
            indent=2,
        )
        + "\n"
    )
