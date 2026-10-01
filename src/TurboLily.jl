"""
    TurboLily

CUDA-only fast path for WaterLily.jl simulations. `TurboLily.Simulation` and
`TurboLily.BiotSimulation` take the same arguments as their WaterLily / BiotSavartBCs
counterparts and return a plain `WaterLily.Simulation`, so `sim_step!`, `measure!`,
`sim_time`, metrics and plotting work unchanged.
"""
module TurboLily

using CUDA, StaticArrays, WaterLily

include("util.jl")
include("flow.jl")
include("poisson.jl")
include("measure.jl")
include("biot.jl")

function Simulation end
"""
    TurboLily.Simulation(dims, uBC, L; kwargs...)

Same signature as `WaterLily.Simulation`; arrays live on the current CUDA device.
"""
Simulation(args...; mem=CuArray, kwargs...) = WaterLily.Simulation(args...; mem=CuArray,
    flow_ctor=(dims,uBC;kw...)->TFlow(dims,uBC;kw...), pois_ctor=TPoisson, kwargs...)

end # module
