using DrWatson
@quickactivate "Ring Attractor"

include(srcdir("stability.jl"))

e0 = 0.0
e1 = 0.2
i0 = 0.0
i1 = 0.2
step = 0.001
noise = 0.5e-3
seed = 2020

r_e = e0:step:e1
r_i = i0:step:i1

m = find_stable_w(r_e, r_i, noise, seed)
noise *= 1e3 
tmp = @dict e0 e1 i0 i1 step noise seed
safesave(datadir("stability", savename(tmp, "csv")), m)
