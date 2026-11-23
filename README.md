# Lower bound C_1c >= 0.410378195

Certificates for the Barnard–Steinerberger autocorrelation constant
([C_1c](https://teorth.github.io/optimizationproblems/constants/1c.html)).

## Claim

    C_1c >= 0.410378195722515456620959705443384843306...

The exact rational is the field `bound` of `pwcert_p20_dps40_S120.json` (55 kB).

## Certificate (piecewise polynomial)

All numbers in the JSON are exact rationals (strings). The function is

    f = (1/s) 1_[-s/2, s/2) + h,    s = 2^-120,

where h = 0 below t1, h(x) = sum_k c_k (x - lo)^k on each piece [lo, hi) listed in `pieces`, and h = 0 beyond the
last piece. There are 11 pieces: 9 polynomials of degree 20 between the breakpoints
{t1, 1 - 5 t1, ..., 1 - t1} (t1 = 22545746012851/2^47 ≈ 0.160197), then the constant T on [1 - t1, 1) and on
[1, 1 + s/2).

Since f is piecewise polynomial with rational data, g(t) = ∫ f(x) f(x+t) dx is a polynomial with rational coefficients
on each piece of the lag mesh, of degree at most 41, and can be computed exactly. The certificate states

    min_{0<=t<=1} g(t) >= m' = T - margin,    ∫ f = 1 + M,

so C_1c >= m' / (1 + M)^2 = `bound`. In fact g >= T on [0, 1], with g = T exactly on [1 - t1 + s/2, 1]
(at 1 - t1 itself, g - T = 4.17e-71 > 0; thanks to chessing234 for the correction).

## Verify

    sha256sum -c SHA256SUMS
    python verify_pwcert_independent.py pwcert_p20_dps40_S120.json   # ~1 min
    python verify_pwcert.py pwcert_p20_dps40_S120.json               # ~15 min

Both verifiers use exact rational arithmetic (Python `Fraction`) for everything that matters and check f >= 0
(Bernstein coefficients per piece), M and `bound`, and g >= m' on [0, 1] (exact Bernstein subdivision on every
lag piece). They compute g differently:

- `verify_pwcert_independent.py` builds g on each lag piece from exact bivariate antiderivatives, and checks that g is
  continuous, that g(0) = ∫ f^2, that 2 ∫ g = (∫ f)^2, and that g vanishes at the end of the support. It was written from
  this description only, without reading `verify_pwcert.py` or the construction.
- `verify_pwcert.py` integrates f(x) f(x+t) exactly at 2p+3 rational lags per piece, interpolates, checks the degree,
  and adds a floating-point FFT sanity check.

## Construction (not needed for validity)

Near-optimal functions have a spike at the left end carrying about 40% of the mass, then f = 0 on (0, t1),
then a nearly flat profile up to x ≈ 1. With every shift in [t1, 1] active, the profile solves
a h(t) + (h ⋆ h)(t) = T on (t1, 1]. Its solution is analytic between the breakpoints {k t1} ∪ {1 - k t1}.

`construction/cont.py` and `construction/contmp.py` solve this equation by piecewise Chebyshev collocation, in double
and in mpmath precision, and maximise over t1. The limit value is 0.41037819572251545662... `construction/pwcert.py`
converts the 40-digit solution to rational polynomials of degree 20 on each piece, takes a dyadic t1 and a box of width
2^-120 for the spike, and certifies the result exactly (`python pwcert.py 40 20 120 --rb 140`, about 6 min).
It is warm-started from the float weights of a K = 2^22 step solution, saved as `K4194304.npy` (from `construction/lean.py`).

The certificate matches the continuum limit of this family to 31 digits. The only loss is from rounding t1 to a dyadic
rational (about 4e-32).

## Earlier step-function certificate

`K67108864_0.41037819.txt.xz` (xz-compressed, 68 MB; 661 MB unpacked): first number K = 2^26, then K + 1
non-negative integer weights, f = w_i on [i/K, (i+1)/K). For equal steps, t -> ∫ f(x) f(x+t) dx is linear between
multiples of 1/K (Kravitz, arXiv:2004.06611, proof of Thm 3.1), so the value is K min_k a_k / S^2 with
a_k = sum_i w_i w_{i+k} and S = sum_i w_i. It gives

    C_1c >= 556819001148976469436453982568448 / 1356843543666599174141709917152655 = 0.410378192642818...

(S = 718053303448502830, minimum at k = 56303821).

    xz -dk K67108864_0.41037819.txt.xz
    python verify_lean.py K67108864_0.41037819.txt      # ~16 min, ~15 GB RAM
    python verify_twopass.py K67108864_0.41037819.txt   # ~12 min, ~15 GB RAM

Both split the weights (up to 2^58) into 8- or 9-bit limbs, so every limb correlation is an integer below 2^52 that
comes out of a float64 FFT within 0.25 (largest distance seen: 0.004), and recombine the limbs exactly in Python
integers. `verify_ntt.py`, `verify_limbs.py`, `verify.py` and `verify_independent.py` are kept for smaller
certificates.

Step functions lose about C/(2K) to the continuum: values at K = 2^20 ... 2^26 are 0.410378000, 0.410378098,
0.410378147, 0.410378171, 0.410378183, 0.410378190, 0.410378193. The single-cell spike with big-integer weights
follows arhancanli's update of teorth/optimizationproblems#344. `construction/` also has the earlier step-function
builds (`one.py`, `lean.py`).
