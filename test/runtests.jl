using NNGraph
# reload("NNGraph.jl")
using Base.Test, Base.tanh

# graph output test setup
m1 = NNGraph.NNMatrix(3,2)
m1.w[1,1] = 1.; m1.w[1,2] = 2.
m1.w[2,1] = 3.; m1.w[2,2] = 4.
m1.w[3,1] = 5.; m1.w[3,2] = 6.

m2 = NNGraph.NNMatrix(2,3)
m2.w[1,1] = 2.; m2.w[1,2] = 3.; m2.w[1,3] = 4.
m2.w[2,1] = 5.; m2.w[2,2] = 6.; m2.w[2,3] = 7.

# add test
g =  NNGraph.Graph()
m3 = NNGraph.add(g,m1,m1)

# size tests
@test m3.n == m1.n
@test m3.d == m1.d
@test size(m3.w,1) == m1.n
@test size(m3.w,2) == m1.d
@test size(m3.dw,1) == m1.n
@test size(m3.dw,2) == m1.d

# forward test
@test m3.w[1,1] == m1.w[1,1] * 2
@test m3.w[2,1] == m1.w[2,1] * 2
@test m3.w[3,1] == m1.w[3,1] * 2
@test m3.w[1,2] == m1.w[1,2] * 2
@test m3.w[2,2] == m1.w[2,2] * 2
@test m3.w[3,2] == m1.w[3,2] * 2


m3.dw[1,1] = .1; m3.dw[1,2] = .2
m3.dw[2,1] = .3; m3.dw[2,2] = .4
m3.dw[3,1] = .5; m3.dw[3,2] = .6

g.backprop[1]()
# set previous chained gradient
@test m1.dw[1,1] == 0.2
@test m1.dw[1,2] == 0.4
@test m1.dw[2,1] == 0.6
@test m1.dw[2,2] == 0.8
@test m1.dw[3,1] == 1.0
@test m1.dw[3,2] == 1.2

# mul test
m1.dw[:] = 0. # reset gradient matrices for next tests
m2.dw[:] = 0. # reset  gradient matrices for next tests
g =  NNGraph.Graph()
m3 = NNGraph.mul(g,m1,m2)
# size tests
@test m3.n == m1.n
@test m3.d == m2.d
@test size(m3.dw,1) == m1.n
@test size(m3.dw,2) == m2.d

# forward tests
@test m3.w[1,1] == 12.
@test m3.w[1,2] == 15.
@test m3.w[1,3] == 18.
@test m3.w[2,1] == 26.
@test m3.w[2,2] == 33.
@test m3.w[2,3] == 40.
@test m3.w[3,1] == 40.
@test m3.w[3,2] == 51.
@test m3.w[3,3] == 62.

m3.dw[1,1] = .1; m3.dw[1,2] = .2; m3.dw[1,3] = .3
m3.dw[2,1] = .4; m3.dw[2,2] = .5; m3.dw[2,3] = .6
m3.dw[3,1] = .7; m3.dw[3,2] = .8; m3.dw[3,3] = .9
g.backprop[1]()

# mul() m1 gradient tests
@test m1.dw[1,1] == 2.
@test_approx_eq_eps m1.dw[1,1] 2.0 1e10
@test_approx_eq_eps m1.dw[1,2] 3.8 1e10
@test_approx_eq_eps m1.dw[2,1] 4.69 1e10
@test_approx_eq_eps m1.dw[2,2] 9.2 1e10
@test_approx_eq_eps m1.dw[3,1] 7.4 1e10
@test_approx_eq_eps m1.dw[3,2] 14.6 1e10

# mul() m2 gradient tests
@test_approx_eq_eps m2.dw[1,1] 4.8 1e10
@test_approx_eq_eps m2.dw[1,2] 5.7 1e10
@test_approx_eq_eps m2.dw[1,3] 6.6 1e10
@test_approx_eq_eps m2.dw[2,1] 5.9 1e10
@test_approx_eq_eps m2.dw[2,2] 7.2 1e10
@test_approx_eq_eps m2.dw[2,3] 8.4 1e10

# reul() tests
m4 = NNGraph.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.
g =  NNGraph.Graph()
m5 = NNGraph.relu(g,m4)

# size tests
@test m5.n == m4.n
@test m5.d == m4.d
@test size(m5.dw,1) == m5.n
@test size(m5.dw,2) == m5.d

@test m5.w[1,1] == 1.
@test m5.w[1,2] == 0.
@test m5.w[2,1] == 0.
@test m5.w[2,2] == 4.
@test m5.w[3,1] == 5.
@test m5.w[3,2] == 0.

m5.dw[1,1] =-.1; m5.dw[1,2] = .2
m5.dw[2,1] = .3; m5.dw[2,2] = .4
m5.dw[3,1] = .5; m5.dw[3,2] = .6

g.backprop[1]()
@test m4.dw[1,1] == -0.1
@test m4.dw[1,2] == 0.
@test m4.dw[2,1] == 0.
@test m4.dw[2,2] == 0.4
@test m4.dw[3,1] == 0.5
@test m4.dw[3,2] == 0.


# rowpluck() tests
m4 = NNGraph.NNMatrix(3,2)
m4.w[1,1] = 1.; m4.w[1,2] =-2.
m4.w[2,1] =-3.; m4.w[2,2] = 4.
m4.w[3,1] = 5.; m4.w[3,2] =-6.

g =  NNGraph.Graph()
m5 = NNGraph.rowpluck(g,m4,2)
@test m5.w[1,1] == -3.
@test m5.w[2,1] == 4.

m5.dw[1,1] =-.1; m5.dw[2,1] = .2
g.backprop[1]()
m4.dw
@test m4.dw[1,1] == 0
@test m4.dw[1,2] == 0
@test m4.dw[2,1] ==-0.1
@test m4.dw[2,2] == 0.2
@test m4.dw[3,1] == 0.
@test m4.dw[3,2] == 0.

# softmax tests
m6 = NNGraph.NNMatrix(5,1)
m6.w[1,1] = 0.3; m6.w[2,1] =0.1; m6.w[3,1] =0.6; m6.w[4,1] = 0.002; m6.w[5,1] = 0.00001
sm = NNGraph.softmax(m6)
@test_approx_eq_eps sm.w[1,1] 0.2149744 1e7
@test_approx_eq_eps sm.w[2,1] 0.1759805 1e7
@test_approx_eq_eps sm.w[3,1] 0.2901595 1e7
@test_approx_eq_eps sm.w[4,1] 0.1595501 1e7
@test_approx_eq_eps sm.w[5,1] 0.1592329 1e7

# tahn() test
g = NNGraph.Graph()
m3 = NNGraph.tanh(g,m1)
# size tests
@test m3.n == m1.n
@test m3.d == m1.d
@test size(m3.w,1) == m1.n
@test size(m3.w,2) == m1.d
@test size(m3.dw,1) == m1.n
@test size(m3.dw,2) == m1.d

# forward tahn tests
@test m3.w[1,1] == tanh(m1.w[1,1])
@test m3.w[1,2] == tanh(m1.w[1,2])
@test m3.w[2,1] == tanh(m1.w[2,1])
@test m3.w[2,2] == tanh(m1.w[2,2])
@test m3.w[3,1] == tanh(m1.w[3,1])
@test m3.w[3,2] == tanh(m1.w[3,2])

# reset gradients then run backprop
m3.dw = ones(size(m3.dw)) * .5
m1.dw = zeros(size(m1.dw))
backprop(g)

# backward tahn tests
@test m1.dw[1,1] == (1.-tanh(m1.w[1,1])^2) *.5
@test m1.dw[1,2] == (1.-tanh(m1.w[1,2])^2) *.5
@test m1.dw[2,1] == (1.-tanh(m1.w[2,1])^2) *.5
@test m1.dw[2,2] == (1.-tanh(m1.w[2,2])^2) *.5
@test m1.dw[3,1] == (1.-tanh(m1.w[3,1])^2) *.5
@test m1.dw[3,2] == (1.-tanh(m1.w[3,2])^2) *.5

# sigmoid() test
g = NNGraph.Graph()
m3 = NNGraph.sigmoid(g,m1)
# size tests
@test m3.n == m1.n
@test m3.d == m1.d
@test size(m3.w,1) == m1.n
@test size(m3.w,2) == m1.d
@test size(m3.dw,1) == m1.n
@test size(m3.dw,2) == m1.d

# forward sigmoid tests
sig(x) = 1.0 / (1.0 + exp(-x))
@test m3.w[1,1] == sig(m1.w[1,1])
@test m3.w[1,2] == sig(m1.w[1,2])
@test m3.w[2,1] == sig(m1.w[2,1])
@test m3.w[2,2] == sig(m1.w[2,2])
@test m3.w[3,1] == sig(m1.w[3,1])
@test m3.w[3,2] == sig(m1.w[3,2])

# reset gradients then run backprop
m3.dw = ones(size(m3.dw)) * .5
m1.dw = zeros(size(m1.dw))
backprop(g)

# backward sigmoid tests
@test m1.dw[1,1] == sig(m1.w[1,1]) * (1.-sig(m1.w[1,1])) *.5
@test m1.dw[1,2] == sig(m1.w[1,2]) * (1.-sig(m1.w[1,2])) *.5
@test m1.dw[2,1] == sig(m1.w[2,1]) * (1.-sig(m1.w[2,1])) *.5
@test m1.dw[2,2] == sig(m1.w[2,2]) * (1.-sig(m1.w[2,2])) *.5
@test m1.dw[3,1] == sig(m1.w[3,1]) * (1.-sig(m1.w[3,1])) *.5
@test m1.dw[3,2] == sig(m1.w[3,2]) * (1.-sig(m1.w[3,2])) *.5

