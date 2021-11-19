export ContinuousMultivariateLogPdf

import DomainSets, DomainIntegrals
import Base: isapprox

"""
    ContinuousMultivariateLogPdf{ D <: DomainSets.Domain, F } <: ContinuousMultivariateDistribution

Generic continuous multivariate dist in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical dist available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package
- `logpdf`: callable object that represents a `logpdf` of a dist. Does not necessarily normalised.

```julia 
fdist = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ 2, (x) -> -x' * x)
```
"""
struct ContinuousMultivariateLogPdf{ D <: DomainSets.Domain, F} <: ContinuousMultivariateDistribution
    domain :: D
    logpdf :: F
end

Base.show(io::IO, ::Type{ <: ContinuousMultivariateLogPdf }) = print(io, "ContinuousMultivariateLogPdf")
Base.show(io::IO, dist::ContinuousMultivariateLogPdf)        = print(io, "ContinuousMultivariateLogPdf()")

ContinuousMultivariateLogPdf(dimension::Int, logpdf::Function) = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ dimension, logpdf)

(dist::ContinuousMultivariateLogPdf)(x::AbstractVector{ <: Real }) = logpdf(dist, x)

Distributions.support(dist::ContinuousMultivariateLogPdf)                           = dist.domain

Distributions.mean(dist::ContinuousMultivariateLogPdf)    = error("mean() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.median(dist::ContinuousMultivariateLogPdf)  = error("median() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.mode(dist::ContinuousMultivariateLogPdf)    = error("mode() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.var(dist::ContinuousMultivariateLogPdf)     = error("var() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.std(dist::ContinuousMultivariateLogPdf)     = error("std() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.cov(dist::ContinuousMultivariateLogPdf)     = error("cov() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.invcov(dist::ContinuousMultivariateLogPdf)  = error("invcov() is not defined for `ContinuousMultivariateLogPdf`.")
Distributions.entropy(dist::ContinuousMultivariateLogPdf) = error("entropy() is not defined for `ContinuousMultivariateLogPdf`.")

Base.ndims(dist::ContinuousMultivariateDistribution) = DomainSets.dimension(dist.domain)

# We don't expect neither `pdf` nor `logpdf` to be normalised
Distributions.pdf(dist::ContinuousMultivariateLogPdf, x::AbstractVector{ <:Real }) = exp(logpdf(dist, x))

function Distributions.logpdf(dist::ContinuousMultivariateLogPdf, x::AbstractVector{ <:Real }) 
    @assert x ∈ dist.domain "x = $(x) does not belong to the domain of $dist"
    return dist.logpdf(x)
end

Base.precision(dist::ContinuousMultivariateLogPdf) = error("precision() is not defined for `ContinuousMultivariateLogPdf`.")

Base.convert(::Type{ ContinuousMultivariateLogPdf }, domain::D, logpdf::F)      where { D <: DomainSets.Domain, F } = ContinuousMultivariateLogPdf{D, F}(domain, logpdf)
Base.convert(::Type{ ContinuousMultivariateLogPdf }, dimension::Int, logpdf::F) where { F }                         = ContinuousMultivariateLogPdf{D, F}(DomainSets().FullSpace() ^ dimension, logpdf)

convert_eltype(::Type{ ContinuousMultivariateLogPdf }, ::Type{ T }, dist::ContinuousMultivariateLogPdf) where { T <: Real } = convert(ContinuousMultivariateLogPdf, dist.domain, dist.logpdf)

vague(::Type{ <: ContinuousMultivariateLogPdf }, dimension::Int) = ContinuousMultivariateLogPdf(DomainSets.FullSpace() ^ dimension, (x) -> 1.0)

prod_analytical_rule(::Type{ <: ContinuousMultivariateLogPdf }, ::Type{ <: ContinuousMultivariateLogPdf }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::ContinuousMultivariateLogPdf, right::ContinuousMultivariateLogPdf)
    @assert left.domain == right.domain "Different domain types in product of generic `ContinuousMultivariateLogPdf` distributions. Left domain is $(left.domain), right is $(right.domain)."
    plogpdf = let left = left, right = right
        (x) -> logpdf(left, x) + logpdf(right, x)
    end
    return ContinuousMultivariateLogPdf(left.domain, plogpdf)
end

# This method is inaccurate and relies on various approximation methods, which may fail in different scenarious
# Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
function Base.isapprox(left::ContinuousMultivariateLogPdf, right::ContinuousMultivariateLogPdf; kwargs...)
    error("TODO: Not implemented")
end


# We do not check typeof of a different functions because in most of the cases lambdas have different types, but it does not really mean that objects are different
function is_typeof_equal(left::ContinuousMultivariateLogPdf{D, F1}, right::ContinuousMultivariateLogPdf{D, F2}) where { D, F1 <: Function, F2 <: Function }
    return true
end