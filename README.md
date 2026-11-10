# Lower bound C_1c >= 0.4103772

Step-function certificate for the Barnard–Steinerberger autocorrelation constant
([C_1c](https://teorth.github.io/optimizationproblems/constants/1c.html)).

## Claim

    C_1c >= 151911829776891904 / 370176031712590489 = 0.410377271251...

## Certificate

`K524288_0.4103772.txt`: first number K = 524288, then 524736 non-negative integer weights.
The function is f = w_i on [i/K, (i+1)/K). Nonzero weights sit at indices 0..524288, so the
support is [0, (K+1)/K).

For equal steps, t -> ∫ f(x) f(x+t) dx is linear between multiples of 1/K
(Kravitz, arXiv:2004.06611, proof of Thm 3.1), so

    min_{0<=t<=1} ∫ f(x)f(x+t) dx / (∫ f)^2 = K * min_{0<=k<=K} a_k / S^2,
    a_k = sum_i w_i w_{i+k},  S = sum_i w_i.

Here S = 2433683732, the minimum is a_k = 4635981133328 at k = 364178.

## Verify

    sha256sum -c SHA256SUMS
    python verify.py K524288_0.4103772.txt               # numpy int64 correlate, ~4 min
    python verify_independent.py K524288_0.4103772.txt   # pure Python big ints, ~1 min

Both compute every lag exactly. `verify.py` checks n * max(w)^2 < 2^62 so the int64 sums
cannot overflow; `verify_independent.py` uses Kronecker substitution and spot-checks lags
by direct summation.

## Construction (not needed for validity)

Near-optimal functions have a spike at the left end carrying about 40% of the mass, then f = 0
on (0, t1) with t1 ≈ 0.168, then a nearly flat profile up to x ≈ 1. With every shift in
[t1, 1] active, the weights satisfy spike * w_k + sum_{i>=1} w_i w_{i+k} = T for all k, which
a damped fixed-point iteration solves in about a hundred FFT steps (`construction/dfp.py`).

The spike is spread over m = 448 cells with geometric weights s_i = M r^i (r = 0.99855). A flat
spike of m cells only resolves f at scale m/K; a geometric one keeps full resolution K, since
its generating polynomial has no zeros on the unit circle. It also carries ~(1-r^m)/(1-r) times
the largest weight, so rounding to integers costs little (about 4e-7 here). Rounding is done by
the same recursion in integers, then a scan over the level T (`construction/shaped.py`,
`mkcert.py`, `vscan.py`).

Values grow like 0.410378 - 0.205/K in K, so this family tops out near 0.410378.

`construction/k524288/` has the K = 524288 build: an Anderson-accelerated fixed point whose
back-substitution is clamped exactly at both edges of the active range (`lib.py`, `mk.py`),
then a small scan over the rounding offset, the scale M and the level T (`lot.py`). Float value
0.4103777, integer certificate 0.4103773. A wider spike (m = 512, r = 0.9987) did not converge.
