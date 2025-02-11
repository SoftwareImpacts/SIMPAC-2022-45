module RulesNormalMixtureOutTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules


@testset "rules:NormalMixture:out" begin

    @testset "Variational : (m_μ::PointMass{ <: Real }..., m_p::PointMass{ <: Real }...)" begin
        
        @test_rules [ with_float_conversions = true ] NormalMixture{2}(:out, Marginalisation) [
            (input = (q_switch = Categorical([ 0.5, 0.5 ]), q_m = (PointMass(1.0), PointMass(1.0)), q_p = (PointMass(1.0), PointMass(1.0))), output = NormalMeanPrecision(1.0, 1.0)),
            (input = (q_switch = Categorical([ 1.0, 0.0 ]), q_m = (PointMass(1.0), PointMass(2.0)), q_p = (PointMass(2.0), PointMass(1.0))), output = NormalMeanPrecision(1.0, 2.0)),
            (input = (q_switch = Categorical([ 0.0, 1.0 ]), q_m = (PointMass(2.0), PointMass(-3.0)), q_p = (PointMass(4.0), PointMass(3.0))), output = NormalMeanPrecision(-3.0, 3.0)),
        ]

    end

    @testset "Variational : (m_μ::UnivariateNormalDistributionsFamily..., m_p::GammaDistributionsFamily...)" begin
        
        @test_rules [ with_float_conversions = true ] NormalMixture{2}(:out, Marginalisation) [
            (input = (q_switch = Categorical([ 0.5, 0.5 ]), q_m = (NormalMeanVariance(1.0, 2.0), NormalMeanPrecision(-2.0, 3.0)), q_p = (GammaShapeRate(1.0, 1.0), GammaShapeScale(2.0, 0.1))), output = NormalMeanPrecision(-1/2, 6/10)),
            (input = (q_switch = Categorical([ 0.75, 0.25 ]), q_m = (NormalWeightedMeanPrecision(-1.0, 2.0), NormalMeanPrecision(2.0, 3.0)), q_p = (GammaShapeScale(1.0, 1.0), GammaShapeRate(2.0, 0.1))), output = NormalMeanPrecision(1/8, 5.75)),
            (input = (q_switch = Categorical([ 1.0, 0.0 ]), q_m = (NormalMeanVariance(1.0, 2.0), NormalMeanPrecision(-2.0, 3.0)), q_p = (GammaShapeRate(1.0, 1.0), GammaShapeScale(2.0, 0.1))), output = NormalMeanPrecision(1, 1)),
        ]

    end

    @testset "Variational : (m_μ::PointMass{ <: Vector }..., m_p::PointMass{ <: Matrix }...)" begin
        
        @test_rules [ with_float_conversions = true ] NormalMixture{2}(:out, Marginalisation) [
            (input = (q_switch = Categorical([ 0.5, 0.5 ]), q_m = (PointMass([ 1.0, 0.0 ]), PointMass([ -1.0, -2.0 ])), q_p = (PointMass(2.0 * diageye(2)), PointMass(3.0 * diageye(2)))), output = MvNormalWeightedMeanPrecision([ -1/2, -3 ], [ 5/2 0; 0 5/2 ])),
            (input = (q_switch = Categorical([ 1.0, 0.0 ]), q_m = (PointMass([ 1.0, 0.0 ]), PointMass([ -1.0, -2.0 ])), q_p = (PointMass(2.0 * diageye(2)), PointMass(3.0 * diageye(2)))), output = MvNormalWeightedMeanPrecision([ 2, 0 ], [ 2 0; 0 2 ])),
            (input = (q_switch = Categorical([ 0.0, 1.0 ]), q_m = (PointMass([ 1.0, 0.0 ]), PointMass([ -1.0, -2.0 ])), q_p = (PointMass(2.0 * diageye(2)), PointMass(3.0 * diageye(2)))), output = MvNormalWeightedMeanPrecision([ -3, -6 ], [ 3 0; 0 3 ])),
        ]

    end

    @testset "Variational : (m_μ::MultivariateNormalDistributionsFamily..., m_p::Wishart...)" begin
        
        @test_rules [ with_float_conversions = true ] NormalMixture{2}(:out, Marginalisation) [
            (input = (q_switch = Categorical([ 0.5, 0.5 ]), q_m = (MvNormalMeanCovariance([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ]), MvNormalMeanPrecision([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ])), q_p = (Wishart(3.0, [ 2.0 -0.25; -0.25 1.0 ]), Wishart(3.0, [ 1.0 -0.25; -0.25 2.0 ]))), output = MvNormalWeightedMeanPrecision([ 6.75, 12.0 ], [ 4.5 -0.75; -0.75 4.5 ])),
            (input = (q_switch = Categorical([ 0.75, 0.25 ]), q_m = (MvNormalWeightedMeanPrecision([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ]), MvNormalMeanPrecision([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ])), q_p = (Wishart(3.0, [ 2.0 -0.25; -0.25 1.0 ]), Wishart(3.0, [ 1.0 -0.25; -0.25 2.0 ]))), output = MvNormalWeightedMeanPrecision([ 3.75, 10.3125 ], [ 5.25 -0.75; -0.75 3.75 ])),
            (input = (q_switch = Categorical([ 0.0, 1.0 ]), q_m = (MvNormalMeanCovariance([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ]), MvNormalWeightedMeanPrecision([ 2.0, 3.0 ], [ 2.0 0.0; 0.0 1.0 ])), q_p = (Wishart(3.0, [ 2.0 -0.25; -0.25 1.0 ]), Wishart(3.0, [ 1.0 -0.25; -0.25 2.0 ]))), output = MvNormalWeightedMeanPrecision([ 0.75, 17.25 ], [ 3.0 -0.75; -0.75 6.0 ])),
        ]

    end

end


end