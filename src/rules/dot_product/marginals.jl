export marginalrule

@marginalrule typeof(dot)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, m_in2::MultivariateNormalDistributionsFamily, meta::Any) = begin
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in1)
    mf_in2 = @call_rule typeof(dot)(:in2, Marginalisation) (m_out = m_out, m_in1 = m_in1, meta = meta)
    # q_in2 = prod(ProdAnalytical(), m_in2, MvNormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    q_in2 = prod(ProdAnalytical(), m_in2, mf_in2)
    # println("="^80)
    # #  @show mean_cov(m_out)
    # @show mean_cov(m_in2), is_initial(messages[3])
    # @show mean_cov(mf_in2)
    # @show entropy(q_in2)
    # println("="^80)
    return (in1 = m_in1, in2 = q_in2)
end
