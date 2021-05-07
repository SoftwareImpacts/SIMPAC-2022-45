export SGCV, SGCVMetadata

import StatsFuns: log2π

struct SGCVMetadata{ A <: AbstractApproximationMethod }
    approximation :: A
end

get_approximation(meta::SGCVMetadata) = meta.approximation

struct SGCV end

@node SGCV Stochastic [ y, x, z, κ, ω , s]

const DefaultSGCVNodeMetadata = SGCVMetadata(GaussHermiteCubature(21))

default_meta(::Type{ SGCV }) = DefaultSGCVNodeMetadata


function ψ(yx)
    m, V = mean(yx),cov(yx)
    (m[1] - m[2])*(m[1] - m[2]) + V[1] + V[4] - V[3] - V[2]
end

function ϕ(z, κ, ω, s)
    ms = probvec(s)
    mω, Vω = mean(ω), cov(ω) 
    mz, vz = mean(z), cov(z)
    mκ, Vκ = mean(κ), cov(κ)
    exp(-ms'*mκ*mz - ms'*mω + 0.5((ms'*mκ)^2*vz + mz^2*ms'*Vκ*ms + ms'*Vκ*ms*vz + ms'*Vω*ms))
end

@average_energy SGCV (q_y_x::MultivariateNormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any, q_s::Any) = begin
    
    m_z, var_z = mean(q_z),cov(q_z)
    m_κ, var_κ = mean(q_κ),cov(q_κ)
    m_ω, var_ω = mean(q_ω),cov(q_ω)
    m_s = probvec(q_s)

    0.5log(2*pi) + 0.5*(m_s'*m_κ*m_z + m_s'*m_ω) + 0.5*(ψ(marg_y_x)*ϕ(marg_z, marg_κ, marg_ω, marg_s))
end