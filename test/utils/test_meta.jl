module ReactiveMPMetaSpecificationHelpers

using Test
using ReactiveMP
using GraphPPL
using Distributions

import ReactiveMP: resolve_meta

@testset "Meta specification" begin

    expr = :(out ~ Normal(args))

    @testset "Use case #1" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta"
        end

        model = Model()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        @test resolve_meta(expr, :SomeOtherNode, (x, y), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x, y), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (y, x), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (x, y, z), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (x, y, x, y), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (x, y, x, y, z, z), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (y, x, z), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (y, z, x), meta, model) == "meta"
        @test resolve_meta(expr, :SomeNode, (x, ), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x, z), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (y, ), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (y, z), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (z, ), meta, model) === nothing

    end

    @testset "Use case #2" begin 

        @meta function makemeta(flag)
            if flag
                SomeNode(x, y) -> "meta1"
            else
                SomeNode(x, y) -> "meta2"
            end
        end

        model = Model()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        for (meta, result) in ((makemeta(true), "meta1"), (makemeta(false), "meta2"))
            @test resolve_meta(expr, :SomeOtherNode, (x, y), meta, model) === nothing
            @test resolve_meta(expr, :SomeNode, (x, y), meta, model) == result
            @test resolve_meta(expr, :SomeNode, (y, x), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (x, y, z), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (x, y, x, y), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (x, y, x, y, z, z), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (y, x, z), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (y, z, x), meta, model) ==  result
            @test resolve_meta(expr, :SomeNode, (x, ), meta, model) === nothing
            @test resolve_meta(expr, :SomeNode, (x, z), meta, model) === nothing
            @test resolve_meta(expr, :SomeNode, (y, ), meta, model) === nothing
            @test resolve_meta(expr, :SomeNode, (y, z), meta, model) === nothing
            @test resolve_meta(expr, :SomeNode, (z, ), meta, model) === nothing
        end

    end

    @testset "Use case #3" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta1"
            SomeNode(z, y) -> "meta2"
        end

        model = Model()

        x = randomvar(model, :x, 10)
        y = randomvar(model, :y, 10)
        z = randomvar(model, :z, 10)

        @test resolve_meta(expr, :SomeNode, (x[1], z[1]), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x[1], z[1], z[2]), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x[1], x[2], z[1]), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x[1], y[1]), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (x[1], x[2], y[1]), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (y[1], y[2], x[1]), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (z[1], z[2], y[1]), meta, model) == "meta2"
        @test resolve_meta(expr, :SomeNode, (y[1], y[2], z[1]), meta, model) == "meta2"
        @test_throws ErrorException resolve_meta(expr, :SomeNode, (x[1], y[1], z[1]), meta, model) 
        @test_throws ErrorException resolve_meta(expr, :SomeNode, (z[1], y[1], x[1]), meta, model) 

    end

    @testset "Use case #4" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta1"
        end

        model = Model()

        x = randomvar(model, :x, 10)
        y = randomvar(model, :y)
        tmp = randomvar(model, :tmp, proxy_variables = (y, ))

        @test resolve_meta(expr, :SomeNode, (x[1], tmp), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (x[1], x[2], tmp), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (tmp, x[1]), meta, model) == "meta1"
        @test resolve_meta(expr, :SomeNode, (tmp, x[1], x[2]), meta, model) == "meta1"

        @test resolve_meta(expr, :SomeNode, (y, ), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (tmp, ), meta, model) === nothing
        for i in 1:10
            @test resolve_meta(expr, :SomeNode, (x[i], ), meta, model) === nothing
            @test resolve_meta(expr, :SomeOtherNode, (x[i], y), meta, model) === nothing
        end

    end

    @testset "Use case #5" begin 

        # Just many variables and statements
        meta = @meta begin
            HGF(x) -> 123 
            AR(x, y, z) -> 1
            AR(x1, y, z) -> 2
            AR(x2, y, z) -> 3
            AR(x3, y, z) -> 4
            AR(x4, y, z) -> 5
            AR(x5, y, z) -> 6
            AR(x6, y, z) -> 7
            AR(x7, y, z) -> 8
            AR(x8, y, z) -> 9
            AR(x9, y, z) -> 10
            AR(x10, y, z) -> 11
            AR(x1, y1, z) -> 12
            AR(x2, y2, z) -> 13
            AR(x3, y3, z) -> 14
            AR(x4, y4, z) -> 15
            AR(x5, y5, z) -> 16
            AR(x6, y6, z3) -> 17
            AR(x7, y7, z4) -> 18
            AR(x8, y8, z) -> 19
            AR(x9, y9, z) -> 20
            AR(x10, y10, z) -> 21
        end

        model = Model()

        z = randomvar(model, :z)
        x = randomvar(model, :x10)
        y = randomvar(model, :y10)
    
        @test ReactiveMP.resolve_meta(expr, :AR, (z, x, y), meta, model) === 21
        @test ReactiveMP.resolve_meta(expr, :AR, (x, z, y), meta, model) === 21
        @test ReactiveMP.resolve_meta(expr, :AR, (x, y, z), meta, model) === 21

    end
    
    # Errors

    @testset "Error case #1" begin 

        meta = @meta begin 
            SomeNode(x, y) -> "meta"
            SomeNode(y, x) -> "meta"
        end

        model = Model()

        x = randomvar(model, :x)
        y = randomvar(model, :y)
        z = randomvar(model, :z)

        @test resolve_meta(expr, :SomeOtherNode, (x, y), meta, model) === nothing
        @test_throws ErrorException resolve_meta(expr, :SomeNode, (x, y), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (y, x), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (x, y, z), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (x, y, x, y), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (x, y, x, y, z, z), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (y, x, z), meta, model)
        @test_throws ErrorException  resolve_meta(expr, :SomeNode, (y, z, x), meta, model)
        @test resolve_meta(expr, :SomeNode, (x, ), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (x, z), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (y, ), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (y, z), meta, model) === nothing
        @test resolve_meta(expr, :SomeNode, (z, ), meta, model) === nothing

    end

end

end