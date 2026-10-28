"""
    Homogeneity

Julia implementation of the Python package **pyHomogeneity**: homogeneity /
change-point tests for univariate time series data.

A homogeneity test checks whether a time series comes from a single
distribution, i.e. it detects one change point (break point) where the
statistical properties of the series change. This is widely used in
hydrology, climatology, finance and quality control.

Available tests (each returns a named tuple with fields `h`, `cp`, `p`,
a test statistic and `avg = (mu1, mu2)`):

* `pettitt_test`          — Pettitt's test (Pettitt, 1979)                                  → statistic `U`
* `snht_test`             — Standard Normal Homogeneity Test (Alexandersson, 1986)          → statistic `T`
* `buishand_q_test`       — Buishand's Q statistics test (Buishand, 1982)                   → statistic `Q`
* `buishand_range_test`   — Buishand's range test (Buishand, 1982)                          → statistic `R`
* `buishand_likelihood_ratio_test` — Buishand's likelihood ratio test (Buishand, 1984)      → statistic `V`
* `buishand_u_test`       — Buishand's U statistics test (Buishand, 1984)                   → statistic `U`

Every test accepts:

* `x`    — univariate data vector (any `AbstractVector`; `missing` and `NaN` values are skipped)
* `alpha`— significance level (default `0.05`)
* `sim`  — number of Monte Carlo simulations for the p-value (default `20000`).
           Pass `nothing` (or `0`) as the third argument to skip the simulation.
           For `pettitt_test` an analytical p-value approximation is used
           instead; for the other tests `h` and `p` are then `missing`.

Output named tuple:

* `h`   — `true` if the series is non-homogeneous, `false` if homogeneous,
          `missing` when no p-value was computed
* `cp`  — probable change-point location (1-based index into the *original* data)
* `p`   — p-value (`missing` when simulation is skipped)
* stat  — test statistic (`U`, `T`, `Q`, `R`, `V` or `U`, depending on the test)
* `avg` — named tuple `(mu1=..., mu2=...)` with the mean before / after `cp`

Direct unpacking works, just like in pyHomogeneity:

    h, cp, p, U, mu = pettitt_test(x, 0.05)

References
----------
1. Pettitt, A. N. (1979). A non-parametric approach to the change-point problem.
   Journal of the Royal Statistical Society: Series C (Applied Statistics),
   28(2), 126–135. doi: 10.2307/2346729

2. Alexandersson, H. (1986). A homogeneity test applied to precipitation data.
   Journal of Climatology, 6(6), 661–675. doi: 10.1002/joc.3370060607

3. Buishand, T. A. (1982). Some methods for testing the homogeneity of rainfall
   records. Journal of Hydrology, 58(1–2), 11–27. doi: 10.1016/0022-1694(82)90066-X

4. Buishand, T. A. (1984). Tests for detecting a shift in the mean of
   hydrological time series. Journal of Hydrology, 73(1–2), 51–69.
   doi: 10.1016/0022-1694(84)90032-5

5. I. Mahmud, S. H. Bari, M. M. Hussain and M. T. Rahman (2015). Homogeneity of
   Rainfall and Temperature Series in Bangladesh. Proceedings of the
   International Conference on Climate Change and Water Security, MIST, Dhaka.
   doi: 10.13140/RG.2.1.4431.3688

6. Pohlert, T. (2016). Package 'trend': Non-Parametric Trend Tests and
   Change-Point Detection.

"""
module Homogeneity

using Statistics: mean, std
using Random: default_rng, randn

export pettitt_test,
       snht_test,
       buishand_q_test,
       buishand_range_test,
       buishand_likelihood_ratio_test,
       buishand_u_test

# ============================================================================
# Supporting computational kernels (explicit loops only, no vectorisation)
# ============================================================================

"""
    __preprocessing(x)

Copy `x` into a plain `Vector{Float64}` and build an index vector that records
the original position (1-based) of every observation.

A single-column matrix is flattened; any other one-dimensional container is
accepted as-is. Missing entries (`missing`) are stored as `NaN` here and
removed later by [`__missing_values_analysis`](@ref).

Returns `(x, c, idx)` where `c` is the number of columns (always `1`) and
`idx[i]` is the original position of `x[i]`.
"""
function __preprocessing(x)
    if x isa AbstractMatrix
        if size(x, 2) == 1
            x = vec(x)
        else
            throw(
                ArgumentError(
                    "Homogeneity tests require a univariate (1-D) data vector.",
                ),
            )
        end
    end

    n = length(x)
    xf = Vector{Float64}(undef, n)
    for i in 1:n
        xi = x[i]
        xf[i] = ismissing(xi) ? NaN : Float64(xi)
    end

    idx = Vector{Int}(undef, n)
    for i in 1:n
        idx[i] = i
    end

    return xf, 1, idx
end

"""
    __missing_values_analysis(x, idx)

Skip every observation that is `missing` or `NaN` (method = 'skip', exactly
as in pyHomogeneity).

Returns `(x_clean, n, idx_clean)` where `n` is the number of remaining
observations and `idx_clean[i]` is the original position of `x_clean[i]` in
the input data.
"""
function __missing_values_analysis(x, idx)
    n = length(x)
    x_clean = Vector{Float64}(undef, n)
    idx_clean = Vector{Int}(undef, n)

    m = 0
    for i in 1:n
        xi = x[i]
        if ismissing(xi) || isnan(xi)
            continue
        end
        m += 1
        x_clean[m] = xi
        idx_clean[m] = idx[i]
    end

    resize!(x_clean, m)
    resize!(idx_clean, m)
    return x_clean, m, idx_clean
end

"""
    rankdata(x)

Assign ranks to the observations of `x` (1 = smallest). Tied values receive
the average of the ranks they would occupy — exactly like
`scipy.stats.rankdata(x, method='average')`.
"""
function rankdata(x::AbstractVector{<:Real})
    n = length(x)
    r = Vector{Float64}(undef, n)
    ord = sortperm(x)  # ord[k] is the index of the k-th smallest value

    i = 1
    while i <= n
        j = i
        while j < n && x[ord[j + 1]] == x[ord[j]]
            j += 1
        end
        avg_rank = (i + j) / 2.0  # average rank of the tied block i..j
        for k in i:j
            r[ord[k]] = avg_rank
        end
        i = j + 1
    end
    return r
end

"""
    maxabs_index(u)

Return `(maximum absolute value, index of its first occurrence)` using a plain
loop — the loop-based counterpart of `abs(u).argmax()`.
"""
function maxabs_index(u::AbstractVector{<:Real})
    best = abs(u[1])
    best_i = 1
    for i in 2:length(u)
        v = abs(u[i])
        if v > best
            best = v
            best_i = i
        end
    end
    return best, best_i
end

"""
    __mean_split(x, loc)

Mean of the observations before and after the change point `loc`:

    mu1 = mean(x[1:loc]),   mu2 = mean(x[loc+1:end])

Returns the named tuple `(mu1=mu1, mu2=mu2)`.
"""
function __mean_split(x, loc)
    n = length(x)

    s1 = 0.0
    for i in 1:loc
        s1 += x[i]
    end
    mu1 = s1 / loc

    cnt = n - loc
    s2 = 0.0
    for i in (loc + 1):n
        s2 += x[i]
    end
    mu2 = cnt == 0 ? NaN : s2 / cnt

    return (mu1 = mu1, mu2 = mu2)
end

# ----------------------------------------------------------------------------
# Test-statistic kernels: each returns (statistic, change-point location)
# ----------------------------------------------------------------------------

"""
    __pettitt(x)

Pettitt's statistic based on ranks. Returns `(U, loc)` where `U` is the
maximum absolute value of the rank-based statistic and `loc` its 1-based
location.
"""
function __pettitt(x)
    n = length(x)
    r = rankdata(x)

    s = Vector{Float64}(undef, n - 1)
    acc = 0.0
    for i in 1:(n - 1)
        acc += r[i]
        s[i] = acc
    end

    U = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        U[i] = 2.0 * s[i] - i * (n + 1)
    end

    return maxabs_index(U)
end

"""
    __snht(x)

Standard Normal Homogeneity Test statistic (Alexandersson, 1986).
Returns `(T, loc)` with `loc` in `1:(n-1)`.
"""
function __snht(x)
    n = length(x)
    xm = mean(x)
    xs = std(x; corrected = true)  # sample standard deviation (ddof = 1)

    # Prefix sums: s[i] = sum(x[1:i]), i = 1..n-1
    s = Vector{Float64}(undef, n - 1)
    acc = 0.0
    for i in 1:(n - 1)
        acc += x[i]
        s[i] = acc
    end

    # Suffix sums: rs[i] = sum(x[i+1:n]), i = 1..n-1
    rs = Vector{Float64}(undef, n - 1)
    acc = 0.0
    for i in n:-1:2
        acc += x[i]
        rs[i - 1] = acc
    end

    T = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        z1 = ((s[i] - i * xm) / xs) / i
        z2 = ((rs[i] - (n - i) * xm) / xs) / (n - i)
        T[i] = i * z1^2 + (n - i) * z2^2
    end

    return maxabs_index(T)
end

"""
    __buishand_partial_sums(x)

Partial sums ``S_k = \\sum_{i=1}^{k} x_i - k \\cdot \\bar{x}`` together with
the population standard deviation (ddof = 0, as in NumPy's `x.std()`).
"""
function __buishand_partial_sums(x)
    n = length(x)
    xm = mean(x)
    xs = std(x; corrected = false)  # population standard deviation (ddof = 0)

    S = Vector{Float64}(undef, n)
    acc = 0.0
    for i in 1:n
        acc += x[i]
        S[i] = acc - i * xm
    end
    return S, xs
end

"""
    __buishand_q(x)

Buishand's Q statistic: ``Q = max_k |S_k| / (s \\sqrt{n})``.
"""
function __buishand_q(x)
    S, xs = __buishand_partial_sums(x)
    n = length(S)
    umax, loc = maxabs_index(S)
    Q = umax / (xs * sqrt(n))
    return Q, loc
end

"""
    __buishand_range(x)

Buishand's range statistic:
``R = (max_k S_k - min_k S_k) / (s \\sqrt{n})``.
"""
function __buishand_range(x)
    S, xs = __buishand_partial_sums(x)
    n = length(S)

    smax = S[1]
    smin = S[1]
    for i in 2:n
        if S[i] > smax
            smax = S[i]
        elseif S[i] < smin
            smin = S[i]
        end
    end

    _, loc = maxabs_index(S)  # location from the absolute partial sums
    R = (smax - smin) / (xs * sqrt(n))
    return R, loc
end

"""
    __buishand_lr(x)

Buishand's likelihood-ratio statistic:
``V = max_k |S_k| / (s \\sqrt{k(n-k)})`` for ``k = 1:(n-1)``.
The change-point location is taken from the full partial-sum vector `S`.
"""
function __buishand_lr(x)
    S, xs = __buishand_partial_sums(x)
    n = length(S)

    V = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        V[i] = S[i] / (xs * sqrt(i * (n - i)))
    end

    vmax, _ = maxabs_index(V)
    _, loc = maxabs_index(S)
    return vmax, loc
end

"""
    __buishand_u(x)

Buishand's U statistic:
``U = \\sum_{k=1}^{n-1} (S_k / s)^2 / (n (n + 1))``.
"""
function __buishand_u(x)
    S, xs = __buishand_partial_sums(x)
    n = length(S)

    acc = 0.0
    for i in 1:(n - 1)
        v = S[i] / xs
        acc += v * v
    end
    U = acc / (n * (n + 1))
    _, loc = maxabs_index(S)
    return U, loc
end

# ----------------------------------------------------------------------------
# Monte Carlo p-value
# ----------------------------------------------------------------------------

"""
    __mc_p_value(func, stat, n, sim, rng)

Estimate the p-value by Monte Carlo simulation: draw `sim` random samples of
size `n` from the standard normal distribution, compute the test statistic of
each sample with `func`, and return the fraction of simulated statistics that
are strictly greater than the observed `stat` — exactly as in pyHomogeneity.
"""
function __mc_p_value(func, stat, n, sim, rng)
    # One reusable buffer for all simulations (avoids `sim` allocations).
    x = Vector{Float64}(undef, n)
    count = 0
    for _ in 1:sim
        for i in 1:n
            x[i] = randn(rng)
        end
        s, _ = func(x)
        if s > stat
            count += 1
        end
    end
    return count / sim
end

# ----------------------------------------------------------------------------
# Common test driver
# ----------------------------------------------------------------------------

"""
    __test(func, x, alpha, sim)

Run a homogeneity-test kernel on `x`, optionally estimate a Monte Carlo
p-value and split the sample mean at the detected change point.

Returns `(h, cp, p, stat, avg, n)` where `n` is the number of valid
(non-missing) observations. When no simulation is requested (`sim ===
nothing` or `sim == 0`), both `h` and `p` are `missing`.
"""
function __test(func, x, alpha, sim)
    x, _, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx)

    stat, loc = func(x)

    if sim === nothing || sim == 0
        p = missing
        h = missing
    else
        p = __mc_p_value(func, stat, n, sim, default_rng())
        h = alpha > p
    end

    avg = __mean_split(x, loc)
    cp = idx[loc]

    return h, cp, p, stat, avg, n
end

# ============================================================================
# Public API
# ============================================================================

"""
    pettitt_test(x, alpha=0.05, sim=20000)

Pettitt's homogeneity test (Pettitt, 1979): a non-parametric test for a
single change point in a time series. It is based on ranks and is
distribution-free.

Arguments
---------
* `x`: univariate data vector (any `AbstractVector`; `missing` and `NaN`
  values are skipped)
* `alpha`: significance level (default `0.05`)
* `sim`: number of Monte Carlo simulations for the p-value (default `20000`).
         Pass `nothing` (or `0`) as the third argument to skip simulation.
         In that case `pettitt_test` still returns the analytical p-value
         approximation

            p = 2 * exp(-6 * U^2 / (n^3 + n^2))

Returns a named tuple `(h=, cp=, p=, U=, avg=)` where `avg = (mu1=, mu2=)`
holds the mean before and after the change point.

Examples
--------
```julia
using Homogeneity

x = rand(1000)
result = pettitt_test(x, 0.05)

# fields can be accessed by name or unpacked directly:
# h, cp, p, U, mu = pettitt_test(x, 0.05)
```
"""
function pettitt_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, U, avg, n = __test(__pettitt, x, alpha, sim)

    # Analytical p-value approximation when no Monte Carlo simulation is run.
    if sim === nothing || sim == 0
        p = 2.0 * exp((-6.0 * U^2) / (n^3 + n^2))
        h = alpha > p
    end

    return (h = h, cp = cp, p = p, U = U, avg = avg)
end

"""
    snht_test(x, alpha=0.05, sim=20000)

Standard Normal Homogeneity Test (SNHT, Alexandersson, 1986): detects a
change in the mean of a normally distributed series by comparing standardized
prefix and suffix sums.

Returns a named tuple `(h=, cp=, p=, T=, avg=)` where `avg = (mu1=, mu2=)`.
"""
function snht_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, T, avg, _ = __test(__snht, x, alpha, sim)
    return (h = h, cp = cp, p = p, T = T, avg = avg)
end

"""
    buishand_q_test(x, alpha=0.05, sim=20000)

Buishand's Q statistics test (Buishand, 1982): based on the maximum absolute
value of the standardized partial sums of the series.

Returns a named tuple `(h=, cp=, p=, Q=, avg=)` where `avg = (mu1=, mu2=)`.
"""
function buishand_q_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, Q, avg, _ = __test(__buishand_q, x, alpha, sim)
    return (h = h, cp = cp, p = p, Q = Q, avg = avg)
end

"""
    buishand_range_test(x, alpha=0.05, sim=20000)

Buishand's range test (Buishand, 1982): based on the range (max minus min)
of the standardized partial sums.

Returns a named tuple `(h=, cp=, p=, R=, avg=)` where `avg = (mu1=, mu2=)`.
"""
function buishand_range_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, R, avg, _ = __test(__buishand_range, x, alpha, sim)
    return (h = h, cp = cp, p = p, R = R, avg = avg)
end

"""
    buishand_likelihood_ratio_test(x, alpha=0.05, sim=20000)

Buishand's likelihood ratio test (Buishand, 1984): based on a weighted,
adjusted version of the partial sums.

Returns a named tuple `(h=, cp=, p=, V=, avg=)` where `avg = (mu1=, mu2=)`.
"""
function buishand_likelihood_ratio_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, V, avg, _ = __test(__buishand_lr, x, alpha, sim)
    return (h = h, cp = cp, p = p, V = V, avg = avg)
end

"""
    buishand_u_test(x, alpha=0.05, sim=20000)

Buishand's U statistics test (Buishand, 1984): based on the sum of squared
standardized partial sums.

Returns a named tuple `(h=, cp=, p=, U=, avg=)` where `avg = (mu1=, mu2=)`.
"""
function buishand_u_test(x, alpha = 0.05, sim = 20000)
    h, cp, p, U, avg, _ = __test(__buishand_u, x, alpha, sim)
    return (h = h, cp = cp, p = p, U = U, avg = avg)
end

end  # module Homogeneity
