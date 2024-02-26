"""
    Simulation

Type representing a simulation object, which can be used to simulate the trajectory of a
dynamical system from an initial condition under a specified control policy.

# Fields
- `t0::Float64`: Initial simulation time - defaults to zero.
- `tf::Float64`: Ending simulation time.
"""
struct Simulation{T <: Real}
    t0::T
    tf::T
    inplace::Bool
end

# Simulation constructor from simulation end time
Simulation(T::Real) = Simulation(0.0, T, true)
Simulation(T::Real, inplace::Bool) = Simulation(0.0, T, inplace)
