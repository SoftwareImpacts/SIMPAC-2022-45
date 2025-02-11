export AbstractMessage, Message, VariationalMessage
export getdata, is_clamped, is_initial, as_message
export multiply_messages

using Distributions
using Rocket

import Rocket: getrecent
import Base: *, +, ndims, precision, length, size, show

"""
    AbstractMessage

An abstract supertype for all concrete message types.

See also: [`Message`](@ref)
"""
abstract type AbstractMessage end

"""
    materialize!(message::AbstractMessage)

Materializes an abstract message and converts it to be of type `Message`.

See also: [`Message`](@ref)
"""
function materialize! end

"""
    Message{D} <: AbstractMessage

`Message` structure encodes a **Belief Propagation** message, which holds some `data` that usually a probability distribution, but can also be an arbitrary object.
Message acts as a proxy structure to `data` object and proxies most of the statistical functions, e.g. `mean`, `mode`, `cov` etc.

# Arguments
- `data::D`: message always holds some data object associated with it
- `is_clamped::Bool`, specifies if this message is clamped
- `is_initial::Bool`, specifies if this message is initial

# Example 

```jldoctest
julia> distribution = Gamma(10.0, 2.0)
Gamma{Float64}(α=10.0, θ=2.0)

julia> message = Message(distribution, false, true)
Message(Gamma{Float64}(α=10.0, θ=2.0))

julia> mean(message) 
20.0

julia> getdata(message)
Gamma{Float64}(α=10.0, θ=2.0)

julia> is_clamped(message)
false

julia> is_initial(message)
true

```

See also: [`AbstractMessage`](@ref), [`materialize!`](@ref)
"""
struct Message{D} <: AbstractMessage
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
end

getdata(message::Message)    = message.data
is_clamped(message::Message) = message.is_clamped
is_initial(message::Message) = message.is_initial

getdata(messages::NTuple{ N, <: Message }) where N = map(getdata, messages)

materialize!(message::Message) = message

Base.show(io::IO, message::Message) = print(io, string("Message(", getdata(message), ")"))

Base.:*(left::Message, right::Message) = multiply_messages(ProdAnalytical(), left, right)

function multiply_messages(prod_parametrisation, left::Message, right::Message) 
    # We propagate clamped message, in case if both are clamped
    is_prod_clamped = is_clamped(left) && is_clamped(right)
    # We propagate initial message, in case if both are initial or left is initial and right is clameped or vice-versa
    is_prod_initial = !is_prod_clamped && (is_clamped_or_initial(left)) && (is_clamped_or_initial(right))

    return Message(prod(prod_parametrisation, getdata(left), getdata(right)), is_prod_clamped, is_prod_initial)
end

# Note: we need extra Base.Generator(as_message, messages) step here, because some of the messages might be VMP messages
# We want to cast it explicitly to a Message structure (which as_message does in case of VariationalMessage)
# We use with Base.Generator to reduce an amount of memory used by this procedure since Generator generates items lazily
prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) = (messages) -> foldl((left, right) -> constrain_form(form_constraint, multiply_messages(prod_constraint, left, right)), Base.Generator(as_message, messages))
prod_foldl_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) = (messages) -> constrain_form(form_constraint, foldl((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)))

prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckEach) = (messages) -> foldr((left, right) -> constrain_form(form_constraint, multiply_messages(prod_constraint, left, right)), Base.Generator(as_message, messages))
prod_foldr_reduce(prod_constraint, form_constraint, ::FormConstraintCheckLast) = (messages) -> constrain_form(form_constraint, foldr((left, right) -> multiply_messages(prod_constraint, left, right), Base.Generator(as_message, messages)))

# Base.:*(m1::Message, m2::Message) = multiply_messages(m1, m2)

Distributions.pdf(message::Message, x)    = Distributions.pdf(getdata(message), x)
Distributions.logpdf(message::Message, x) = Distributions.logpdf(getdata(message), x)

MacroHelpers.@proxy_methods Message getdata [
    Distributions.mean,
    Distributions.median,
    Distributions.mode,
    Distributions.shape,
    Distributions.scale,
    Distributions.rate,
    Distributions.var,
    Distributions.std,
    Distributions.cov,
    Distributions.invcov,
    Distributions.logdetcov,
    Distributions.entropy,
    Distributions.params,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
    Base.eltype,
    mean_cov, 
    mean_var,
    mean_invcov, 
    mean_precision, 
    weightedmean_cov, 
    weightedmean_var,
    weightedmean_invcov, 
    weightedmean_precision,
    probvec,
    weightedmean
]

Distributions.mean(fn::Function, message::Message) = mean(fn, getdata(message))

## Variational Message

mutable struct VariationalMessage{R, S, F} <: AbstractMessage
    messages   :: R
    marginals  :: S
    mappingFn  :: F
    cache      :: Union{Nothing, Message}
end

VariationalMessage(messages::R, marginals::S, mappingFn::F) where { R, S, F } = VariationalMessage(messages, marginals, mappingFn, nothing)

Base.show(io::IO, ::VariationalMessage) = print(io, string("VariationalMessage(:postponed)"))

getcache(vmessage::VariationalMessage)                    = vmessage.cache
setcache!(vmessage::VariationalMessage, message::Message) = vmessage.cache = message

compute_message(vmessage::VariationalMessage) = vmessage.mappingFn((vmessage.messages, getrecent(vmessage.marginals)))

function materialize!(vmessage::VariationalMessage)
    cache = getcache(vmessage)
    if cache !== nothing
        return cache
    end
    message = compute_message(vmessage)
    setcache!(vmessage, message)
    return message
end

## Utility functions

as_message(message::Message)             = message
as_message(vmessage::VariationalMessage) = materialize!(vmessage)

## Operators

# TODO
reduce_messages(messages) = mapreduce(as_message, (left, right) -> multiply_messages(ProdAnalytical(), left, right), messages)

## Message observable 

struct MessageObservable{M <: AbstractMessage} <: Subscribable{M}
    subject :: Rocket.RecentSubjectInstance{M, Subject{M, AsapScheduler, AsapScheduler}}
    stream  :: LazyObservable{M}
end

MessageObservable(::Type{M} = AbstractMessage) where M = MessageObservable{M}(RecentSubject(M), lazy(M))   

function as_message_observable(observable)
    output = MessageObservable(eltype(observable))
    connect!(output, observable)
    return output
end

Rocket.getrecent(observable::MessageObservable) = Rocket.getrecent(observable.subject)

@inline Rocket.on_subscribe!(observable::MessageObservable, actor) = subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Actor{ <: AbstractMessage })           = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.NextActor{ <: AbstractMessage })       = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ErrorActor{ <: AbstractMessage })      = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.CompletionActor{ <: AbstractMessage }) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.Subject{ <: AbstractMessage })                 = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.BehaviorSubjectInstance{ <: AbstractMessage }) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.PendingSubjectInstance{ <: AbstractMessage })  = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.RecentSubjectInstance{ <: AbstractMessage })   = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MessageObservable, actor::Rocket.ReplaySubjectInstance{ <: AbstractMessage })   = Rocket.on_subscribe!(observable.stream, actor)

function connect!(message::MessageObservable, source)
    set!(message.stream, source |> multicast(message.subject) |> ref_count())
    return nothing
end

function setmessage!(message::MessageObservable, value)
    next!(message.subject, Message(value, false, true))
    return nothing
end

## Message Mapping structure
## https://github.com/JuliaLang/julia/issues/42559
## Explanation: Julia cannot fully infer type of the lambda callback function in activate! method in node.jl file
## We create a lambda-like callable structure to improve type inference and make it more stable
## However it is not fully inferrable due to dynamic tags and variable constraints, but still better than just a raw lambda callback

struct MessageMapping{F, T, C, N, M, A, R}
    vtag            :: T
    vconstraint     :: C
    msgs_names      :: N
    marginals_names :: M
    meta            :: A
    factornode      :: R
end

message_mapping_fform(::MessageMapping{F}) where F = F
message_mapping_fform(::MessageMapping{F}) where F <: Function = F.instance

function MessageMapping(::Type{F}, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, factornode::R) where { F, T, C, N, M, A, R }
    return MessageMapping{F, T, C, N, M, A, R}(vtag, vconstraint, msgs_names, marginals_names, meta, factornode)
end

function MessageMapping(::F, vtag::T, vconstraint::C, msgs_names::N, marginals_names::M, meta::A, factornode::R) where { F <: Function, T, C, N, M, A, R} 
    return MessageMapping{F, T, C, N, M, A, R}(vtag, vconstraint, msgs_names, marginals_names, meta, factornode)
end

function (mapping::MessageMapping)(dependencies)

    messages  = dependencies[1]
    marginals = dependencies[2] # getrecent(marginals) call happens in VariationalMessage

    # Message is clamped if all of the inputs are clamped
    is_message_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)

    # Message is initial if it is not clamped and all of the inputs are either clamped or initial
    is_message_initial = !is_message_clamped && (__check_all(is_clamped_or_initial, messages) && __check_all(is_clamped_or_initial, marginals))

    message = rule(
        message_mapping_fform(mapping), 
        mapping.vtag, 
        mapping.vconstraint, 
        mapping.msgs_names, 
        messages, 
        mapping.marginals_names, 
        marginals, 
        mapping.meta, 
        mapping.factornode
    )

    return Message(message, is_message_clamped, is_message_initial)
end

Base.map(::Type{T}, mapping::M) where { T, M <: MessageMapping } = Rocket.MapOperator{T, M}(mapping)


