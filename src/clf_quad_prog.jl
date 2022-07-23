"""
    CLFQuadProg <: FeedbackController

Control Lyapunov Function (CLF)-based quadratic program (QP) to compute control inputs
for a control affine system

# Fields
- `solve` : function that solves the QP for the control input
- `H` : quadratic weight in QP objective
- `F` : linear weight in QP objective
"""
struct CLFQuadProg <: FeedbackController
    solve::Function
    H
    F
end

(k::CLFQuadProg)(x) = k.solve(x)

function CLFQuadProg(Σ::ControlAffineSystem, CLF::ControlLyapunovFunction)
    # Set parameters for objective function
    H = Σ.m == 1 ? 0.0 : zeros(Σ.m, Σ.m)
    F = Σ.m == 1 ? 0.0 : zeros(Σ.m)

    # Construct quadratic program
    function solve(x)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Compute Lie derivatives
        LfV = drift_lie_derivative(CLF, Σ, x)
        LgV = control_lie_derivative(CLF, Σ, x)
        γ = CLF.α(CLF.V(x))

        # Check if we're relaxing the CLF constraint
        if CLF.relax
            @variable(model, δ)
            @constraint(model, LfV + LgV*u <= -γ + δ)
            @objective(model, Min, 0.5*u'*H*u + F'*u + CLF.p*δ^2)
        else
            @constraint(model, LfV + LgV*u <= -γ)
            @objective(model, Min, 0.5*u'*H*u + F'*u)
        end

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return CLFQuadProg(solve, H, F)
end

function CLFQuadProg(Σ::ControlAffineSystem, CLF::ControlLyapunovFunction, H, F)

    # Construct quadratic program
    function solve(x)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Compute Lie derivatives
        LfV = drift_lie_derivative(CLF, Σ, x)
        LgV = control_lie_derivative(CLF, Σ, x)
        γ = CLF.α(CLF.V(x))

        # Check if we're relaxing the CLF constraint
        if CLF.relax
            @variable(model, δ)
            @constraint(model, LfV + LgV*u <= -γ + δ)
            @objective(model, Min, 0.5*u'*H*u + F'*u + CLF.p*δ^2)
        else
            @constraint(model, LfV + LgV*u <= -γ)
            @objective(model, Min, 0.5*u'*H*u + F'*u)
        end

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return CLFQuadProg(solve, H, F)
end