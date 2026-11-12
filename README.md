# Lower bound C_1c >= 0.41037819

Step-function certificate for the Barnard–Steinerberger autocorrelation constant
([C_1c](https://teorth.github.io/optimizationproblems/constants/1c.html)).

## Claim

    C_1c >= 556819001148976469436453982568448 / 1356843543666599174141709917152655 = 0.410378192642818...

## Certificate

`K67108864_0.41037819.txt.xz` (xz-compressed, 68 MB; 661 MB unpacked): first number
K = 67108864 = 2^26, then 67108865 non-negative integer weights. The function is f = w_i on [i/K, (i+1)/K), so the support is [0, (K+1)/K).

For equal steps, t -> ∫ f(x) f(x+t) dx is linear between multiples of 1/K
(Kravitz, arXiv:2004.06611, proof of Thm 3.1), so

    min_{0<=t<=1} ∫ f(x)f(x+t) dx / (∫ f)^2 = K * min_{0<=k<=K} a_k / S^2,
    a_k = sum_i w_i w_{i+k},  S = sum_i w_i.

Here S = 718053303448502830, the minimum is a_k = 3152954882928893840102143785 at k = 56303821.

## Verify

    xz -dk K67108864_0.41037819.txt.xz
    sha256sum -c SHA256SUMS
    python verify_lean.py K67108864_0.41037819.txt      # ~16 min, ~15 GB RAM
    python verify_twopass.py K67108864_0.41037819.txt   # ~12 min, ~15 GB RAM

(times on a 12-core cloud machine). The weights go up to 2^58, past the int64 products of
`verify.py`. Both verifiers split each weight into 8- or 9-bit limbs; every limb correlation is an
integer below 2^52, so a float64 FFT gives it to within 0.25 (checked; the largest distance seen is
0.004), and the exact integers are recombined in Python. They then differ:

- `verify_lean.py` assembles every a_k exactly and takes the minimum.
- `verify_twopass.py` first finds every lag within a relative 1e-12 of the smallest (the float
  error is below 1e-14), then rebuilds those lags exactly.

Both check S, a_0 and the deciding lag by direct summation in Python integers, and both give the
fraction above. `verify_ntt.py` (no floating point) and `verify_limbs.py` do the same job but need
more memory at this size; `verify.py` and `verify_independent.py` are kept for smaller certificates.

## Construction (not needed for validity)

Near-optimal functions have a spike at the left end carrying about 40% of the mass, then f = 0
on (0, t1) with t1 ≈ 0.160, then a nearly flat profile up to x ≈ 1. With every shift in
[t1, 1] active, the weights satisfy spike * w_k + sum_{i>=1} w_i w_{i+k} = T for all k.

The spike is a single cell. The system is solved in floating point by an Anderson-accelerated
fixed point, each K warm-started from K/2 (`construction/lean.py`, about 20 FFT steps per level),
then every weight is rounded at scale 2^58. Rounding at 2^52, as in the earlier K = 2^24
certificate, costs about 4e-10; at 2^58 the cost is about 1e-11. The single-cell spike with
big-integer weights follows arhancanli's update of teorth/optimizationproblems#344.

Values at K = 2^20 ... 2^26: 0.410378000, 0.410378098, 0.410378147, 0.410378171, 0.410378183,
0.410378190, 0.410378193. The continuum limit of this family is 0.41037819572 (computed directly,
piecewise on the breakpoints of the profile), and K (limit - value) ≈ 0.2052, i.e. half the limit,
so the K = 2^26 value is within 3.1e-9 of it.

`construction/` also has the earlier builds (`one.py`; geometric spike over m cells kept within
int64, K = 262144 and K = 524288).
