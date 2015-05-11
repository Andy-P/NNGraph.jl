module NNGraph
import Base.tanh
export NNMatrix, randNNMat, forwardprop, softmax, Solver, step
export Graph, backprop, rowpluck

include("recurrent.jl")
include("graph.jl")
include("solver.jl")


end # module
