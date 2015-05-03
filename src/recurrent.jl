abstract Model # this is either an LSTM or RNN

type NNMatrix # Neural net layer's weights & gradients
    n::Int
    d::Int
    w::Array{Float64,2} # matix of weights
    dw::Array{Float64,2} # matix of gtadients
    NNMatrix(n::Int) = new(n, 1, zeros(n), zeros(n))
    NNMatrix(n::Int, d::Int) = new(n, d, zeros(n,d), zeros(n,d))
    NNMatrix(n::Int, d::Int, w::Array, dw::Array) = new(n, d, w, dw)
end

randNNMat(n::Int, d::Int, std::FloatingPoint) = NNMatrix(n, d, randn(n,d)*std, zeros(n,d))


function softmax(m::NNMatrix)
    out = NNMatrix(m.n,m.d)
    maxval = maximum(m.w)
    out.w[:] = exp(m.w - maxval)
    out.w /= sum(out.w)
    return out
end


