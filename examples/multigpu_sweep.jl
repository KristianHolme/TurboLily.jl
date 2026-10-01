# Parameter sweep: one TurboLily simulation per GPU, driven from one Julia process
# (start Julia with at least as many threads as GPUs, e.g. `julia -t 4`).
using WaterLily, TurboLily, CUDA, StaticArrays, Printf
function sphere(Re; N=64, T=Float32)
    dims = (5N÷2, N, N); R = N÷6
    c = SVector{3,T}(N÷2, N÷2, N÷2); r = T(R)
    TurboLily.BiotSimulation(dims, (1,0,0), 2R; ν=2R/Re, T, body=AutoBody((x,t)->√sum(abs2,x .- c)-r))
end
Res = [100, 300, 1000, 3700]
ndev = length(CUDA.devices())
t = @elapsed results = fetch.([Threads.@spawn begin
    CUDA.device!((k-1) % ndev)
    sim = sphere(Re)
    sim_step!(sim, 10; remeasure=false)
    (Re, -2WaterLily.pressure_force(sim)[1]/(π*(64÷6)^2), CUDA.device())
end for (k, Re) in enumerate(Res)])
for (Re, cd, dev) in results; @printf "Re=%5d  Cd(t=10)=%.3f  on %s\n" Re cd dev; end
@printf "wall time for %d simulations on %d GPUs: %.1f s\n" length(Res) ndev t
