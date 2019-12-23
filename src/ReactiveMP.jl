module ReactiveMP

include("lazy_observable.jl")
include("message.jl")
include("interface.jl")
include("variable.jl")

include("nodes/node.jl")
include("nodes/gaussian.jl")
include("nodes/addition.jl")

end
