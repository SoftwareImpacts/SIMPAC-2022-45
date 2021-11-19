module ContinuousMultivariateLogPdfTest

using Test
using ReactiveMP
using Distributions
using Random

import DomainSets

@testset "Generic Functional Distributions" begin

    @testset "ContinuousMultivariateLogPdf" begin
        
        @testset "Constructor" begin 
            f  = (x) -> -x' * x
            d1 = ContinuousMultivariateLogPdf(2, f)
            d2 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, f)
            d3 = ContinuousMultivariateLogPdf(4, f)
            d4 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 4, f)

            @test typeof(d1) === typeof(d2)
            @test ndims(d1) === 2
            @test ndims(d2) === 2
            @test ndims(d3) === 4
            @test ndims(d4) === 4
            @test_broken d1 ≈ d2
            @test_broken d3 ≈ d4
        end

        @testset "pdf/logpdf" begin 
            d1 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> -x' * x)
            
            f32_points1 = range(Float32(-10.0), Float32(10.0), length = 10)
            f64_points1 = range(-10.0, 10.0, length = 10)
            bf_points1  = range(BigFloat(-10.0), BigFloat(10.0), length = 10)

            f32_grid1 = vec(collect.(collect(Iterators.product(f32_points1, f32_points1))))
            f64_grid1 = vec(collect.(collect(Iterators.product(f64_points1, f64_points1))))
            bf_grid1 = vec(collect.(collect(Iterators.product(bf_points1, bf_points1))))

            points1 = vcat(f32_grid1, f64_grid1, bf_grid1)

            @test all(map(p -> -p' * p == d1(p), points1))
            @test all(map(p -> -p' * p == logpdf(d1, p), points1))
            @test all(map(p -> exp(-p' * p) == pdf(d1, p), points1))

            d2 = ContinuousMultivariateLogPdf(DomainSets.HalfLine() ^ 2, (x) -> -x' * x / 4)
            
            f32_points2 = range(Float32(0.0), Float32(10.0), length = 10)
            f64_points2 = range(0.0, 10.0, length = 10)
            bf_points2  = range(BigFloat(0.0), BigFloat(10.0), length = 10)

            f32_grid2 = vec(collect.(collect(Iterators.product(f32_points2, f32_points2))))
            f64_grid2 = vec(collect.(collect(Iterators.product(f64_points2, f64_points2))))
            bf_grid2 = vec(collect.(collect(Iterators.product(bf_points2, bf_points2))))

            points2 = vcat(f32_grid2, f64_grid2, bf_grid2)

            @test all(map(p -> -p' * p / 4 == d2(p), points2))
            @test all(map(p -> -p' * p / 4 == logpdf(d2, p), points2))
            @test all(map(p -> exp(-p' * p / 4) == pdf(d2, p), points2))

            @test_throws MethodError d2(-1.0)
            @test_throws MethodError logpdf(d2, -1.0)
            @test_throws MethodError pdf(d2, -1.0)
            @test_throws AssertionError d2([ -1.0 ])
            @test_throws AssertionError logpdf(d2, [ -1.0 ])
            @test_throws AssertionError pdf(d2, [ -1.0 ])

            @test_throws MethodError d2(Float32(-1.0))
            @test_throws MethodError logpdf(d2, Float32(-1.0))
            @test_throws MethodError pdf(d2, Float32(-1.0))
            @test_throws AssertionError d2([ Float32(-1.0) ])
            @test_throws AssertionError logpdf(d2, [ Float32(-1.0) ])
            @test_throws AssertionError pdf(d2, [ Float32(-1.0) ])

            @test_throws MethodError d2(BigFloat(-1.0))
            @test_throws MethodError logpdf(d2, BigFloat(-1.0))
            @test_throws MethodError pdf(d2, BigFloat(-1.0))
            @test_throws AssertionError d2([ BigFloat(-1.0) ])
            @test_throws AssertionError logpdf(d2, [ BigFloat(-1.0) ])
            @test_throws AssertionError pdf(d2, [ BigFloat(-1.0) ])

            d3 = ContinuousMultivariateLogPdf(DomainSets.FullSpace(Float32) ^ 2, (x) -> -x' * x / 3)

            @test all(map(p -> -p' * p / 3 == d3(p), points1))
            @test all(map(p -> -p' * p / 3 == logpdf(d3, p), points1))
            @test all(map(p -> exp(-p' * p / 3) == pdf(d3, p), points1))

            d4 = ContinuousMultivariateLogPdf(DomainSets.FullSpace(BigFloat) ^ 2, (x) -> -x' * x / 5)

            @test all(map(p -> -p' * p / 5 == d4(p), points1))
            @test all(map(p -> -p' * p / 5 == logpdf(d4, p), points1))
            @test all(map(p -> exp(-p' * p / 5) == pdf(d4, p), points1))

            d5 = ContinuousMultivariateLogPdf(DomainSets.HalfLine{Float32}() ^ 2, (x) -> -x' * x / 6)

            @test all(map(p -> -p' * p / 6 == d5(p), points2))
            @test all(map(p -> -p' * p / 6 == logpdf(d5, p), points2))
            @test all(map(p -> exp(-p' * p / 6) == pdf(d5, p), points2))

            d6 = ContinuousMultivariateLogPdf(DomainSets.HalfLine{BigFloat}() ^ 2, (x) -> -x' * x / 7)

            @test all(map(p -> -p' * p / 7 == d6(p), points2))
            @test all(map(p -> -p' * p / 7 == logpdf(d6, p), points2))
            @test all(map(p -> exp(-p' * p / 7) == pdf(d6, p), points2))
        end

        @testset "support" begin 
            d1 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> 1.0)
            @test DomainSets.infimum(support(d1)) == [ -Inf, -Inf ]
            @test DomainSets.supremum(support(d1)) == [ Inf, Inf ]

            d2 = ContinuousMultivariateLogPdf(DomainSets.HalfLine() ^ 3, (x) -> 1.0)

            @test DomainSets.infimum(support(d2)) == [ 0.0, 0.0, 0.0 ]
            @test DomainSets.supremum(support(d2)) == [ Inf, Inf, Inf ]
        end

        @testset "vague" begin
            d = vague(ContinuousMultivariateLogPdf, 3)

            @test typeof(d) <: ContinuousMultivariateLogPdf
            @test_broken d ≈ ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 3, (x) -> 1.0)
        end

        @testset "prod" begin
            d1 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> 2.0 * -x' * x)
            d2 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> 3.0 * -x' * x)

            pr1 = prod(ProdAnalytical(), d1, d2)
            pt1 = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> logpdf(d1, x) + logpdf(d2, x))

            @test_broken isapprox(pr1, pt1, atol = 1e-12)

            d3 = ContinuousMultivariateLogPdf(DomainSets.HalfLine() ^ 2, (x) -> 2.0 * -x' * x)
            d4 = ContinuousMultivariateLogPdf(DomainSets.HalfLine() ^ 2, (x) -> 3.0 * -x' * x)

            pr2 = prod(ProdAnalytical(), d3, d4)
            pt2 = ContinuousMultivariateLogPdf(DomainSets.HalfLine() ^ 3, (x) -> logpdf(d3, x) + logpdf(d4, x))

            @test_broken isapprox(pr2, pt2, atol = 1e-12)

            @test_broken !isapprox(pr1, pr2, atol = 1e-12)
        end

        @testset "convert" begin
            d = DomainSets.FullSpace() ^ 2
            l = (x) -> 1.0

            c = convert(ContinuousMultivariateLogPdf, d, l)
            @test typeof(c) <: ContinuousMultivariateLogPdf
            @test_broken isapprox(c, ContinuousMultivariateLogPdf(d, l), atol = 1e-12)

            c2 = convert(ContinuousMultivariateLogPdf, c)
            @test typeof(c2) <: ContinuousMultivariateLogPdf
            @test_broken isapprox(c2, ContinuousMultivariateLogPdf(d, l), atol = 1e-12)

        end

    end

end

end
