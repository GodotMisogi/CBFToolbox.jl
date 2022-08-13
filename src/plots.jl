"""
    plot_vector_field(xs, ys, Σ::ControlAffineSystem; scale=0.15, lw=1)
    plot_vector_field(xs, ys, Σ::ControlAffineSystem, k::FeedbackController; scale=0.15, lw=1)

Plot the vector field of a open or closed-loop `ControlAffineSystem`.
"""
function plot_vector_field(xs, ys, Σ::ControlAffineSystem; scale=0.15, lw=1)
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_vector_field(xs, ys, f, scale=scale, lw=lw)
end

function plot_vector_field(xs, ys, Σ::ControlAffineSystem, k::FeedbackController; scale=0.15, lw=1)
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_vector_field(xs, ys, f, scale=scale, lw=lw)
end

"""
    plot_vector_field!(xs, ys, Σ::ControlAffineSystem; scale=0.15, lw=1)
    plot_vector_field!(xs, ys, Σ::ControlAffineSystem, k::FeedbackController; scale=0.15, lw=1)

Plot the vector field of a open or closed-loop `ControlAffineSystem` on an existing figure.
"""
function plot_vector_field!(xs, ys, Σ::ControlAffineSystem; scale=0.15, lw=1)
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_vector_field!(xs, ys, f, scale=scale, lw=lw)
end

function  plot_vector_field!(xs, ys, Σ::ControlAffineSystem, k::FeedbackController; scale=0.15, lw=1)
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_vector_field!(xs, ys, f, scale=scale, lw=lw)
end

"""
    plot_phase_portrait(X0::Vector{Vector{Float64}}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait(X0::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait(X0::Vector{Vector{Float64}}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait(X0::Vector{Float64}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait(xs::Vector{Float64}, ys::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait(xs::Vector{Float64}, ys::Vector{Float64}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait(xs::Union{StepRangeLen, LinRange}, ys::Union{StepRangeLen, LinRange}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait(xs::Union{StepRangeLen, LinRange}, ys::Union{StepRangeLen, LinRange}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)

Plot the phase portraint of a control affine system.
"""
function plot_phase_portrait(
    X0::Vector{Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_phase_portrait(X0, f, T, lw=lw)
end

function plot_phase_portrait(X0::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    return plot_phase_portrait([X0], Σ, T, lw=lw)
end

function plot_phase_portrait(
    X0::Vector{Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_phase_portrait(X0, f, T, lw=lw)
end

function plot_phase_portrait(
    X0::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    return plot_phase_portrait([X0], Σ, k, T, lw=lw)
end

function plot_phase_portrait(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_phase_portrait(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_phase_portrait(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait(
    xs::Union{StepRangeLen, LinRange}, 
    ys::Union{StepRangeLen, LinRange}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])
    return VectorFieldPlots.plot_phase_portrait(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait(
    xs::Union{StepRangeLen, LinRange}, 
    ys::Union{StepRangeLen, LinRange}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController,
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])
    return VectorFieldPlots.plot_phase_portrait(xs, ys, f, T, lw=lw)
end

"""
    plot_phase_portrait!(X0::Vector{Vector{Float64}}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait!(X0::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait!(X0::Vector{Vector{Float64}}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait!(X0::Vector{Float64}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait!(xs::Vector{Float64}, ys::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait!(xs::Vector{Float64}, ys::Vector{Float64}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)
    plot_phase_portrait!(xs::Union{StepRangeLen, LinRange}, ys::Union{StepRangeLen, LinRange}, Σ::ControlAffineSystem, T::Float64; lw=1)
    plot_phase_portrait!(xs::Union{StepRangeLen, LinRange}, ys::Union{StepRangeLen, LinRange}, Σ::ControlAffineSystem, k::FeedbackController, T::Float64; lw=1)

Plot the phase portraint of a control affine system.
"""
function plot_phase_portrait!(
    X0::Vector{Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_phase_portrait!(X0, f, T, lw=lw)
end

function plot_phase_portrait!(X0::Vector{Float64}, Σ::ControlAffineSystem, T::Float64; lw=1)
    return plot_phase_portrait!([X0], Σ, T, lw=lw)
end

function plot_phase_portrait!(
    X0::Vector{Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_phase_portrait!(X0, f, T, lw=lw)
end

function plot_phase_portrait!(
    X0::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    return plot_phase_portrait!([X0], Σ, k, T, lw=lw)
end

function plot_phase_portrait!(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])

    return VectorFieldPlots.plot_phase_portrait!(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait!(
    xs::Vector{Float64}, 
    ys::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])

    return VectorFieldPlots.plot_phase_portrait!(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait!(
    xs::Union{StepRangeLen, LinRange}, 
    ys::Union{StepRangeLen, LinRange}, 
    Σ::ControlAffineSystem, 
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2])
    return VectorFieldPlots.plot_phase_portrait!(xs, ys, f, T, lw=lw)
end

function plot_phase_portrait!(
    xs::Union{StepRangeLen, LinRange}, 
    ys::Union{StepRangeLen, LinRange}, 
    Σ::ControlAffineSystem, 
    k::FeedbackController,
    T::Float64; 
    lw=1
    )
    f(x1, x2) = Σ.f([x1, x2]) + Σ.g([x1, x2])*k([x1, x2])
    return VectorFieldPlots.plot_phase_portrait!(xs, ys, f, T, lw=lw)
end

"""
    plot_circle(x, y, r; samples=500, lw=1.0, c=:black, fillalpha=0.2)
    plot_circle!(x, y, r; samples=500, lw=1.0, c=:gray70, fillalpha=1.0)

Plot a circle with center (x,y) and radius r.
"""
function plot_circle(x, y, r; samples=500, lw=1.0, c=:black, fillalpha=0.2)
    xs, ys = circle_shape(x, y, r, samples=samples)
    fig = plot(xs, ys, seriestype=[:shape], lw=lw, c=c, linecolor=:black, fillalpha=fillalpha)

    return fig
end

function plot_circle!(x, y, r; samples=500, lw=1.0, c=:gray70, fillalpha=1.0)
    xs, ys = circle_shape(x, y, r, samples=samples)
    fig = Plots.current()
    plot!(fig, xs, ys, seriestype=[:shape], lw=lw, c=c, linecolor=:black, fillalpha=fillalpha)
    
    return fig
end

"""
    circle_shape(x, y, r; samples=500)

Create coordinates of circle for plotting.
"""
function circle_shape(x, y, r; samples=500)
    θ = LinRange(0.0, 2*π, samples)
    xs = x .+ r*cos.(θ)
    ys = y .+ r*sin.(θ)

    return xs, ys
end