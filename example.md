# Examples for Homogeneity.jl

See the main **[README.md](README.md)** for installation and API details.

The examples below need `Homogeneity.jl` plus, for the DataFrames examples,
`DataFrames.jl` and `CSV.jl`:

```julia
using Pkg
Pkg.add(url="https://github.com/mmhs013/Homogeneity.jl.git")
Pkg.add(["DataFrames", "CSV"])
```

```julia
using Homogeneity, DataFrames, CSV, Random, Statistics
```

## Example 1 — Basic usage on a synthetic series

```julia
using Homogeneity, Random

Random.seed!(1)   # for reproducibility

# A homogeneous series (no change point)
x = randn(200)
res = pettitt_test(x, 0.05)
println(res)
# (h = false, cp = ..., p = ..., U = ..., avg = (mu1 = ..., mu2 = ...))

# A series with an artificial shift in the mean at observation 101
y = vcat(randn(100), randn(100) .+ 3.0)
res_y = pettitt_test(y, 0.05)
println(res_y.cp)   # 100: last observation before the shift
println(res_y.h)    # true: non-homogeneous
println(res_y.avg.mu1, "  ", res_y.avg.mu2)  # means before / after cp
```

## Example 2 — All six tests on the same data

The classic pyHomogeneity sample series (360 observations, including missing
values) is analysed with every test. Missing values are skipped automatically:

```julia
using Homogeneity

# pyHomogeneity's sample data (360 values with NaNs)
sample = Float64[
    32., 20., 25., 189., 240., 193., 379., 278., 301., 0., 0.,
    82., 0., 4., NaN, NaN, 121., 234., 360., 262., 120., 30.,
    11., 1., 7., 3., 31., 31., 355., 102., 248., 274., 308.,
    NaN, 5., 26., 11., 16., 6., 48., 388., 539., 431., 272.,
    404., 186., 0., 2., 0., 4., 1., 54., 272., 459., 235.,
    164., 365., 135., 2., NaN, NaN, 4., 0., 128., 210., 163.,
    446., 225., 462., 467., 19., 13., 0., 3., 17., 132., 178.,
    338., 525., 623., 145., 31., 19., 3., 0., 29., 25., 87.,
    259., 756., 486., 180., 292., 43., 92., 1., 0., 16., 2.,
    0., 130., 253., 594., 111., 273., 30., 0., 4., 0., 27.,
    24., 41., 292., 378., 499., 265., 320., 227., 4., 0., 4.,
    14., 8., 48., 416., 240., 404., 207., 733., 105., 0., 112.,
    0., 14., 0., 30., 140., 202., 289., 159., 424., 106., 3.,
    0., 65., 3., 14., 58., 268., 466., 432., 266., 240., 95.,
    1., 0., 10., 26., 4., 114., 94., 289., 173., 208., 263.,
    156., 5., 0., 16., 16., 14., 0., 111., 475., 534., 432.,
    471., 117., 70., 1., 3., 28., 7., 401., 184., 283., 338.,
    171., 335., 176., 0., 0., 10., 11., 9., 140., 102., 208.,
    298., 245., 220., 29., 2., 27., 10., 13., 26., 84., 143.,
    367., 749., 563., 283., 353., 10., 0., 0., 0., 0., 9.,
    246., 265., 343., 429., 168., 133., 17., 0., 18., 35., 76.,
    158., 272., 250., 190., 289., 466., 84., 0., 0., 0., 0.,
    0., 22., 217., 299., 185., 115., 344., 203., 8., NaN, NaN,
    0., 5., 284., 123., 254., 476., 496., 326., 27., 20., 0.,
    4., 53., 72., 113., 214., 364., 219., 220., 156., 264., 0.,
    13., 0., 0., 45., 90., 137., 638., 529., 261., 206., 251.,
    0., 0., 5., 9., 58., 72., 138., 130., 471., 328., 356.,
    523., 0., 1., 0., 0., 12., 143., 193., 184., 192., 138.,
    174., 69., 1., 0., 0., 18., 25., 28., 92., 732., 320.,
    256., 302., 131., 15., 0., 27., 0., 22., 20., 213., 393.,
    474., 374., 109., 159., 0., 0., 0., 3., 3., 49., 205.,
    128., 194., 570., 169., 89., 0., 0., 0., 0., 0., 26.,
    185., 286., 92., 225., 244., 190., 3., 20.]

tests = [
    ("Pettitt",          pettitt_test(sample, 0.05, nothing)),
    ("SNHT",             snht_test(sample, 0.05, nothing)),
    ("Buishand Q",       buishand_q_test(sample, 0.05, nothing)),
    ("Buishand Range",   buishand_range_test(sample, 0.05, nothing)),
    ("Buishand LR",      buishand_likelihood_ratio_test(sample, 0.05, nothing)),
    ("Buishand U",       buishand_u_test(sample, 0.05, nothing)),
]

# Simple table of the results
for (name, res) in tests
    stat_val = begin
        if name == "Pettitt" || name == "Buishand U"; res.U
        elseif name == "SNHT"; res.T
        elseif name == "Buishand Q"; res.Q
        elseif name == "Buishand Range"; res.R
        elseif name == "Buishand LR"; res.V
        end
    end
    println(rpad(name, 17), rpad(string(res.cp), 6),
            rpad(round(stat_val; digits=6), 12),
            rpad(string(res.h === missing ? "-" : res.h), 8),
            res.p === missing ? "-" : round(res.p; digits=6))
end

# Expected (from pyHomogeneity):
# Test              cp    statistic    h      p
# Pettitt           298   2716.0       false  0.733289
# SNHT              298   2.442659     -      -
# Buishand Q        298   0.595546     -      -
# Buishand Range    298   0.989316     -      -
# Buishand LR       298   0.083303     -      -
# Buishand U        298   0.064404     -      -
```

All change points are located at observation 298 of the original series, and
the means before/after the change point are `mu1 ≈ 157.87` and `mu2 ≈ 120.94`,
exactly as in pyHomogeneity.

## Example 3 — DataFrames + CSV (real-world workflow)

Homogeneity tests work on plain vectors, so they slot straight into a
DataFrames workflow. Here we write a small CSV file, read it with `CSV.jl`,
turn a column into a `DataFrame`, run a homogeneity test on that column and
store the result back in the data frame:

```julia
using Homogeneity, DataFrames, CSV, Random

# ---- 1. Create (or point to) a CSV file ------------------------------
Random.seed!(42)
df = DataFrame(
    station = repeat(["A", "B"], inner=100),
    year    = repeat(1901:2000, outer=2),
    # station A is homogeneous; station B shifts from mean 10 to mean 13
    # halfway through its record (observation 51)
    value   = vcat(randn(100) .+ 10.0,
                   randn(50) .+ 10.0, randn(50) .+ 13.0),
)
CSV.write("homogeneity_sample.csv", df)

# ---- 2. Read it back ------------------------------------------------
df = CSV.read("homogeneity_sample.csv", DataFrame)

# ---- 3. Test each station separately --------------------------------
results = DataFrame(station = String[], cp = Int[], p = Float64[],
                    statistic = Float64[],
                    homogeneous = Union{Missing, Bool}[])

for station in unique(df.station)
    x = filter(row -> row.station == station, df).value   # Vector{Float64}
    res = pettitt_test(x, 0.05, 5000)                     # fewer sims for speed
    push!(results, (station=station, cp=res.cp,
                    p=res.p, statistic=res.U,
                    homogeneous = res.h === missing ? missing : !res.h))
end

println(results)
# station  cp  p        statistic  homogeneous
# A        ... ...      ...        true
# B        50  ~0.0     ...        false
```

Because every test returns a named tuple, the fields can be pulled out while
building the summary table (`res.cp`, `res.p`, `res.U`, `res.h`).

## Example 4 — Change-point means before / after the break

The `avg` field of the result holds the mean of the segment before and after
the detected change point — useful for quantifying the size of the break:

```julia
using Homogeneity, Random

Random.seed!(7)
x = vcat(randn(150) .+ 5.0, randn(150) .- 2.0)   # mean drops from 5 to -2

res = snht_test(x, 0.05, 2000)
println("change point: ", res.cp)
println("mean before : ", res.avg.mu1)  # ≈ 5
println("mean after  : ", res.avg.mu2)  # ≈ -2
println("p-value     : ", res.p)
```

## Notes

* `20000` is the default number of simulations and matches pyHomogeneity, but
  for quick exploration a smaller number (e.g. `2000`) is often enough.
* Pass `nothing` as the third argument to skip Monte Carlo simulation.
  `pettitt_test` then reports the analytical p-value approximation; the other
  tests report `h = missing` and `p = missing`.
* All numeric results agree with pyHomogeneity to the last significant digit
  (see `test/runtests.jl` for the exact reference values).
