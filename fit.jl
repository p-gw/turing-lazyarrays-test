using CategoricalArrays, CSV, DataFrames, LazyArrays, Memoization, ReverseDiff, Turing
using StatsFuns: logistic

url = "https://raw.githubusercontent.com/t-alfers/Rasch-Turing/master/dichotom.txt"
data = CSV.read(download(url), DataFrame, delim = "\t")
data = data[:, Not([:Anger, :Sex])]
data.person = collect(1:nrow(data))

data_long = DataFrames.stack(data, Not(:person), :person)
y = data_long.value
p = data_long.person
i = levelcode.(categorical(data_long.variable))

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function lazyarray(f, x)::BroadcastArray
  LazyArray(Base.broadcasted(f, x))
end

@model function irt(y, i, p; I = maximum(i), P = maximum(p))
  theta ~ filldist(Normal(), P)
  mu ~ Normal()
  sigma ~ truncated(Normal(), 0, Inf)
  beta ~ filldist(Normal(mu, sigma), I)

  y ~ arraydist(lazyarray(x -> Bernoulli(logistic(x)), theta[p] .- beta[i]))
end

model = irt(y, i, p)

Turing.sample(model, Turing.HMC(0.05, 10), 2000)
Turing.sample(model, Turing.NUTS(1000, 0.65), 2000)
