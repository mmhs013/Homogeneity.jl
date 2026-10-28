using Homogeneity
using Test
using Random

# ---------------------------------------------------------------------------
# Sample data: identical to pyHomogeneity's tests/test_pyhomogeneity.py
# (360 observations with missing values, as used by pyHomogeneity's unit tests)
# ---------------------------------------------------------------------------
const SAMPLE_DATA = Float64[
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

# Deterministic statistics match pyHomogeneity to the last significant digit
# (observed agreement is ~1e-16, i.e. machine precision).
approx(a, b) = isapprox(a, b; rtol = 1e-14, atol = 0.0)

@testset "Homogeneity.jl" begin

    # ------------------------------------------------------------------
    # Reference values from pyHomogeneity (tests/test_pyhomogeneity.py)
    # ------------------------------------------------------------------
    @testset "Pettitt test (no simulation -> analytical p-value)" begin
        res = pettitt_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test res.U == 2716.0
        @test approx(res.p, 0.7332886357063501)
        @test res.h == false
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)

        # sim = 0 behaves like no simulation
        @test pettitt_test(SAMPLE_DATA, 0.05, 0).p == res.p
        @test pettitt_test(SAMPLE_DATA, 0.05, 0).h == res.h

        # Direct unpacking works, just like in pyHomogeneity
        h, cp, p, U, mu = pettitt_test(SAMPLE_DATA, 0.05, nothing)
        @test h == false && cp == 298 && U == 2716.0
        @test approx(mu.mu1, 157.87285223367698)
    end

    @testset "SNHT test (no simulation)" begin
        res = snht_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test approx(res.T, 2.4426594259172947)
        @test res.h === missing
        @test res.p === missing
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)
    end

    @testset "Buishand Q test (no simulation)" begin
        res = buishand_q_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test approx(res.Q, 0.5955457285563376)
        @test res.h === missing
        @test res.p === missing
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)
    end

    @testset "Buishand range test (no simulation)" begin
        res = buishand_range_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test approx(res.R, 0.9893156056266303)
        @test res.h === missing
        @test res.p === missing
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)
    end

    @testset "Buishand likelihood ratio test (no simulation)" begin
        res = buishand_likelihood_ratio_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test approx(res.V, 0.08330290132452312)
        @test res.h === missing
        @test res.p === missing
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)
    end

    @testset "Buishand U test (no simulation)" begin
        res = buishand_u_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298
        @test approx(res.U, 0.0644043126990563)
        @test res.h === missing
        @test res.p === missing
        @test approx(res.avg.mu1, 157.87285223367698)
        @test approx(res.avg.mu2, 120.93548387096774)
    end

    # ------------------------------------------------------------------
    # Monte Carlo p-values: all six tests
    # ------------------------------------------------------------------
    @testset "Monte Carlo p-values (all six tests)" begin
        all_tests = [
            ("Pettitt", pettitt_test),
            ("SNHT", snht_test),
            ("Buishand Q", buishand_q_test),
            ("Buishand Range", buishand_range_test),
            ("Buishand LR", buishand_likelihood_ratio_test),
            ("Buishand U", buishand_u_test),
        ]
        for (name, f) in all_tests
            Random.seed!(2024)
            res = f(SAMPLE_DATA, 0.05, 2000)
            @test res.cp == 298
            @test res.p isa Float64
            @test (0.0 <= res.p) && (res.p <= 1.0)
            @test res.h isa Bool
        end

        # Seeded runs are reproducible
        Random.seed!(1234)
        p1 = pettitt_test(SAMPLE_DATA, 0.05, 500).p
        Random.seed!(1234)
        p2 = pettitt_test(SAMPLE_DATA, 0.05, 500).p
        @test p1 == p2
    end

    @testset "Non-homogeneous series is detected (h = true)" begin
        Random.seed!(7)
        x = vcat(randn(100), randn(100) .+ 3.0)   # mean shift at observation 101
        res = pettitt_test(x, 0.05, 2000)
        @test res.cp == 100
        @test res.h == true
        @test res.p < 0.05
    end

    # ------------------------------------------------------------------
    # Missing values: NaN and missing must be handled identically
    # ------------------------------------------------------------------
    @testset "Missing values are skipped (NaN and missing)" begin
        # Sample data contains 7 NaN values; only 353 valid ones are used,
        # and cp = 298 refers to a position in the *original* data.
        res = pettitt_test(SAMPLE_DATA, 0.05, nothing)
        @test res.cp == 298

        # NaN-free data gives the same statistic
        clean = filter(!isnan, SAMPLE_DATA)
        @test pettitt_test(clean, 0.05, nothing).U == 2716.0

        # `missing` and `NaN` must produce identical results
        Random.seed!(11)
        x = vcat(randn(50), randn(50) .+ 3.0)
        x_nan = copy(x); x_nan[5] = NaN
        x_mis = Vector{Union{Missing, Float64}}(x); x_mis[5] = missing
        r_nan = pettitt_test(x_nan, 0.05, nothing)
        r_mis = pettitt_test(x_mis, 0.05, nothing)
        @test r_nan.cp == r_mis.cp
        @test r_nan.U == r_mis.U
        @test approx(r_nan.avg.mu1, r_mis.avg.mu1)
        @test approx(r_nan.avg.mu2, r_mis.avg.mu2)
    end

    # ------------------------------------------------------------------
    # Degenerate inputs: parity with pyHomogeneity
    # ------------------------------------------------------------------
    @testset "Constant series" begin
        res = pettitt_test(ones(10), 0.05, nothing)
        @test res.cp == 1
        @test res.U == 0.0
        @test res.p == 2.0          # analytical approximation is not clamped
        @test res.h == false

        for f in (snht_test, buishand_q_test, buishand_range_test,
                  buishand_likelihood_ratio_test, buishand_u_test)
            r = f(ones(10), 0.05, nothing)
            @test r.cp == 1
            @test r.h === missing
            @test r.p === missing
        end
        @test isnan(snht_test(ones(10), 0.05, nothing).T)
        @test isnan(buishand_q_test(ones(10), 0.05, nothing).Q)
        @test isnan(buishand_range_test(ones(10), 0.05, nothing).R)
        @test isnan(buishand_likelihood_ratio_test(ones(10), 0.05, nothing).V)
        @test isnan(buishand_u_test(ones(10), 0.05, nothing).U)
    end

    @testset "Very short series (n = 1 and n = 2)" begin
        # n = 1: Buishand Q/Range/U are defined (as in pyHomogeneity)
        @test buishand_q_test([1.0], 0.05, nothing).cp == 1
        @test isnan(buishand_q_test([1.0], 0.05, nothing).Q)
        @test isnan(buishand_range_test([1.0], 0.05, nothing).R)
        @test buishand_u_test([1.0], 0.05, nothing).U == 0.0
        # n = 1: Pettitt/SNHT/LR cannot be computed (same as pyHomogeneity)
        @test_throws Exception pettitt_test([1.0], 0.05, nothing)
        @test_throws Exception snht_test([1.0], 0.05, nothing)
        @test_throws Exception buishand_likelihood_ratio_test([1.0], 0.05, nothing)

        # n = 2: reference values verified against pyHomogeneity
        @test approx(pettitt_test([1.0, 2.0], 0.05, nothing).p, 1.2130613194252668)
        @test approx(snht_test([1.0, 2.0], 0.05, nothing).T, 0.9999999999999998)
        @test approx(buishand_q_test([1.0, 2.0], 0.05, nothing).Q, 0.7071067811865475)
        @test approx(buishand_range_test([1.0, 2.0], 0.05, nothing).R, 0.7071067811865475)
        @test approx(buishand_likelihood_ratio_test([1.0, 2.0], 0.05, nothing).V, 1.0)
        @test approx(buishand_u_test([1.0, 2.0], 0.05, nothing).U, 0.16666666666666666)
    end

    @testset "Empty and all-missing inputs raise an error" begin
        @test_throws Exception pettitt_test(Float64[], 0.05, nothing)
        @test_throws Exception pettitt_test([missing, missing], 0.05, nothing)
        @test_throws Exception snht_test(Float64[], 0.05, nothing)
    end

    # ------------------------------------------------------------------
    # Input flexibility
    # ------------------------------------------------------------------
    @testset "Input flexibility" begin
        x = collect(1.0:10.0)
        @test pettitt_test(x).cp > 0
        @test pettitt_test(reshape(x, 10, 1)).cp == pettitt_test(x).cp
        @test pettitt_test(view(x, 1:5)).cp == pettitt_test(x[1:5]).cp
        @test pettitt_test(collect(1:10)).cp == pettitt_test(x).cp   # Int vector
        @test pettitt_test((1.0, 2.0, 3.0, 4.0)).cp == pettitt_test([1.0, 2.0, 3.0, 4.0]).cp

        # alpha is honoured
        @test snht_test(x, 0.01, nothing).cp == snht_test(x, 0.05, nothing).cp

        # Multi-column input is rejected
        @test_throws ArgumentError pettitt_test(ones(5, 2), 0.05, nothing)
    end

end  # testset

println("All Homogeneity.jl tests passed.")
