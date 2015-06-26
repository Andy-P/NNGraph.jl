
type NNMatrix # Neural net layer's weights & gradients
    n::Int
    d::Int
    w::Matrix{Float64} # matrix of weights
    dw::Matrix{Float64} # matrix of gradients
    rnd::Matrix{Float64} # matrix of random nums (empty except when using dropout)
    NNMatrix(n::Int) = new(n, 1, zeros(n,1), zeros(n,1),zeros(0,0))
    NNMatrix(n::Int, d::Int) = new(n, d, zeros(n,d), zeros(n,d), zeros(0,0))
    NNMatrix(n::Int, d::Int, w::Matrix, dw::Matrix) = new(n, d, w, dw, zeros(0,0))
    NNMatrix(n::Int, d::Int, w::Matrix, dw::Matrix, rnd::Matrix) = new(n, d, w, dw, rnd)
    NNMatrix(w::Matrix) = new(size(w,1), size(w,2), w, zeros(size(w,1),size(w,2)), zeros(0,0))
end

randNNMat(n::Int, d::Int, std::FloatingPoint=1.) = NNMatrix(n, d, randn(n,d)*std, zeros(n,d))

#----- loss functions ------

function softmax(m::NNMatrix)
    out = NNMatrix(m.n,m.d)
    maxval = maximum(m.w)
    out.w[:] = exp(m.w - maxval)
    out.w /= sum(out.w)
    return out
end
