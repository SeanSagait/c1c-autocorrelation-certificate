# Lower bound C_1c >= 0.4103781

Step-function certificate for the Barnard–Steinerberger autocorrelation constant
([C_1c](https://teorth.github.io/optimizationproblems/constants/1c.html)).

## Claim

    C_1c >= 679700908027579442984780824576 / 1656279491328950142501183554731 = 0.410378146675...

## Certificate

`K4194304_0.4103781.txt`: first number K = 4194304 = 2^22, then 4194305 non-negative integer
weights. The function is f = w_i on [i/K, (i+1)/K), so the support is [0, (K+1)/K).

For equal steps, t -> ∫ f(x) f(x+t) dx is linear between multiples of 1/K
(Kravitz, arXiv:2004.06611, proof of Thm 3.1), so

    min_{0<=t<=1} ∫ f(x)f(x+t) dx / (∫ f)^2 = K * min_{0<=k<=K} a_k / S^2,
    a_k = sum_i w_i w_{i+k},  S = sum_i w_i.

Here S = 11219502722536334, the minimum is a_k = 12316052677654275337897144 at k = 3513048.

## Verify

    sha256sum -c SHA256SUMS
    python verify_ntt.py K4194304_0.4103781.txt     # integers only, ~7 min
    python verify_limbs.py K4194304_0.4103781.txt   # float FFT on 9-bit limbs, ~1 min

The weights go up to 2^52, past the int64 range of `verify.py`, so two other verifiers are
included. Both compute every lag and check a_0 and the deciding lag by direct summation.

- `verify_ntt.py` uses no floating point: it computes all a_k modulo five NTT primes and
  rebuilds them by the Chinese remainder theorem, valid since a_k <= n max(w)^2 is below the
  product of the primes.
- `verify_limbs.py` splits each weight into 9-bit limbs; each limb correlation is below 2^53,
  so a float64 FFT gives it to within 0.25 (checked), and the limbs are recombined in Python
  integers.

`verify.py` and `verify_independent.py` (Kronecker substitution, exact but slow at this size)
are kept for smaller certificates.

## Construction (not needed for validity)

Near-optimal functions have a spike at the left end carrying about 40% of the mass, then f = 0
on (0, t1) with t1 ≈ 0.168, then a nearly flat profile up to x ≈ 1. With every shift in
[t1, 1] active, the weights satisfy spike * w_k + sum_{i>=1} w_i w_{i+k} = T for all k.

Here the spike is a single cell, w_0 = 2^52, and the system is solved in floating point by an
Anderson-accelerated fixed point (about 30 FFT steps), then every weight is rounded at scale
2^52 (`construction/one.py`). With weights this large rounding costs about 1e-13. The
single-cell spike with big-integer weights follows arhancanli's update of
teorth/optimizationproblems#344.

Values at K = 2^20, 2^21, 2^22: 0.410377995, 0.410378086, 0.410378147. They are still
increasing; the limit of this family is not known.

`construction/` also has the earlier builds (geometric spike over m cells kept within int64,
K = 262144 and K = 524288).
