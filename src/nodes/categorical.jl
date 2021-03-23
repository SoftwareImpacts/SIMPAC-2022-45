
@node Categorical Stochastic [ out, p ]

@average_energy Categorical (q_out::Categorical, q_p::Any) = -sum(probvec(q_out) .* logmean(q_p))