module NNGraph

import Base.tanh

export NNMatrix, randNNMat, softmax
export Graph, backprop, rowpluck, tanh, sigmoid, relu, mul, add, eltmul, dot
export Solver, step

include("nnMatrix.jl")
include("graph.jl")
include("solver.jl")


end # module
