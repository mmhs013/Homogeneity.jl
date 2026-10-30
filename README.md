# Homogeneity.jl

[![CI](https://github.com/mmhs013/Homogeneity.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mmhs013/Homogeneity.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**Homogeneity.jl** is a Julia implementation of the Python package
**[pyHomogeneity](https://github.com/mmhs013/pyHomogeneity)**: homogeneity /
change-point tests for univariate time series data.

A homogeneity test checks whether a time series comes from a single
distribution, i.e. it detects one change point (break point) where the
statistical properties of the series change. This is an essential test in
hydrology, climatology, finance, quality control and many other fields where
long series are analysed.

## What is the Homogeneity Test ?

The homogeneity test is a statistical test method that checks whether two (or
more) datasets come from the same distribution. In a time series it is applied
to detect one (or more) change/breakpoint — the point where the data set
changes its distribution. Lots of statistical analyses require a homogeneous
dataset, which makes this test important in statistical analysis.

`Homogeneity.jl` performs six commonly used homogeneity tests:

1. **Pettitt's test** (`pettitt_test`)
2. **Standard Normal Homogeneity Test (SNHT)** (`snht_test`)
3. **Buishand's Q test** (`buishand_q_test`)
4. **Buishand's Range Test** (`buishand_range_test`)
5. **Buishand's Likelihood Ratio Test** (`buishand_likelihood_ratio_test`)
6. **Buishand's U Test** (`buishand_u_test`)

## Function details

All homogeneity test functions accept almost the same input arguments:

- **x**: a vector of data (any Julia `AbstractVector`; `missing` and `NaN`
  values are skipped automatically)
- **alpha**: significance level (default `0.05`)
- **sim**: number of Monte Carlo simulations for the p-value calculation
  (default `20000`). Pass `nothing` (or `0`) as the third argument to skip
  simulation. For `pettitt_test` an analytical p-value approximation is used
  in that case; for the other tests `h` and `p` are then `missing`.

All homogeneity tests return a named tuple which contains:

- **h**: `true` if the data is non-homogeneous, `false` if homogeneous,
  `missing` when no p-value was computed
- **cp**: probable change point location (1-based index into the *original*
  data)
- **p**: p-value of the significance test (`missing` when simulation is skipped)
- **U / T / Q / R / V**: test statistic (depends on the test method)
- **avg**: mean values before and after the change point, as `(mu1=..., mu2=...)`

Results can be accessed by name or unpacked directly, just like in pyHomogeneity:

```julia
using Homogeneity

x = rand(1000)

result = pettitt_test(x, 0.05)
print(result)                          # named tuple
print(result.cp)                       # access by name
# or unpack directly:
h, cp, p, U, mu = pettitt_test(x, 0.05)
```

## Design note

pyHomogeneity relies on NumPy vectorisation. In this Julia port all
computational kernels are written with **explicit loops instead of
vectorisation**, because plain loops are fast and idiomatic in Julia and
avoid the allocation overhead of broadcasted temporaries.

## Dependencies

`Homogeneity.jl` only needs the Julia standard libraries:

- [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/)
- [Random](https://docs.julialang.org/en/v1/stdlib/Random/)

It requires **Julia 1.3 or later**.

## Installation

Install from the General registry (available once the registration pull
request is merged):

```julia
julia> using Pkg
julia> Pkg.add("Homogeneity")
```

To use the current development version directly from GitHub:

```julia
julia> Pkg.add(url="https://github.com/mmhs013/Homogeneity.jl.git")
```

## Tests

The package is unit-tested against the reference results of pyHomogeneity:

```julia
julia> Pkg.test("Homogeneity")
```

## Usage

A quick example:

```julia
using Homogeneity

# Data generation for analysis
data = rand(360)

result = pettitt_test(data)
println(result)
```

Output is like this:

```julia
(h = false, cp = 89, p = 0.1428..., U = 3811.0, avg = (mu1 = 0.5487..., mu2 = 0.4688...))
```

The output is a named tuple, so the user can access any component by name:

```julia
println(result.cp)
println(result.avg.mu1)
```

Several worked examples — including how to combine `Homogeneity.jl` with
[data-science packages such as DataFrames.jl and CSV.jl](example.md) — are
provided in **[example.md](example.md)**.

## Contributions

`Homogeneity.jl` is a community project and welcomes contributions.

## Code of Conduct

`Homogeneity.jl` wishes to maintain a positive community.

## License

MIT

## References

1. Alexandersson, H., 1986. A homogeneity test applied to precipitation data.
   Journal of climatology, 6(6), pp.661-675. doi: [10.1002/joc.3370060607](https://doi.org/10.1002/joc.3370060607)

2. Buishand, T.A., 1982. Some methods for testing the homogeneity of rainfall
   records. Journal of hydrology, 58(1-2), pp.11-27. doi: [10.1016/0022-1694(82)90066-X](https://doi.org/10.1016/0022-1694(82)90066-X)

3. Buishand, T.A., 1984. Tests for detecting a shift in the mean of hydrological
   time series. Journal of hydrology, 73(1-2), pp.51-69.
   doi: [10.1016/0022-1694(84)90032-5](https://doi.org/10.1016/0022-1694(84)90032-5)

4. I. Mahmud, S. H. Bari, M. M. Hussain and M. T. Rahman (2015), Homogeneity of
   Rainfall and Temparature Series in Bangladesh, Proceedings of the
   International Conference on Climate Change and Water Security, Held in
   December 27, 2015, MIST, Dhaka, Bangladesh.
   doi: [10.13140/RG.2.1.4431.3688](https://doi.org/10.13140/RG.2.1.4431.3688)

5. Pettitt, A.N., 1979. A non-parametric approach to the change-point problem.
   Journal of the Royal Statistical Society: Series C (Applied Statistics),
   28(2), pp.126-135. doi: [10.2307/2346729](https://doi.org/10.2307/2346729)

6. Pohlert, T., 2016. Package 'trend'. [Title Non-Parametric Trend Tests and
   Change-Point Detection](https://cran.r-project.org/web/packages/trend/vignettes/trend.pdf).
