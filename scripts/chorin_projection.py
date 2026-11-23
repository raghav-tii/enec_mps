#!/usr/bin/env python3
"""Chorin projection for 2D incompressible Navier-Stokes in MPS/QTT form.

The module avoids three pitfalls of a naive MPS projection scheme, each of which makes the
run diverge within ~30 steps on the ENEC rod bundle.  All three were measured, not inferred:

1. The Poisson right-hand side must be the divergence of the predictor that is about to be
   corrected, recomputed every step.  Reusing the divergence of an earlier (e.g. the corrected)
   field projects one field using another field's divergence.

2. The outlet Dirichlet correction added to the right-hand side must be reset every step.  If it
   accumulates, 33% of |b| is added per step and 10.3x by step 30 on the bundle, which matches
   when the run actually blows up.

3. The mask is applied *after* the projection, which destroys the thing the
   projection just established.  div(m*u) = m*div(u) + u.grad(m), and the
   second term does not vanish: starting from a discretely divergence-free
   field with |div u| = 1.1e-07, masking gives |div(m*u)| = 1.35e+05, and
   |u.grad(m)| = 1.35e+05 -- the entire residual, to three digits.

The fix for (1) and (2) is to compute the right-hand side fresh each step.
The fix for (3) is to mask the *predictor* and project afterwards, so the
projection removes whatever divergence the mask introduced.  That is volume
penalisation with the penalty parameter equal to dt; the rods are then
enforced to O(dt) slip instead of exactly, which is the standard trade and is
what makes the scheme stable.

Run `python3 chorin_projection.py` for the test suite.
"""

import contextlib
import os

import numpy as np
import torch

import tensnet as tt

torch.set_default_dtype(torch.float64)

# The MPO builders read tensnet's global qubit ordering; shuffle_axis encoding
# only matches them under "sequential".  See bundle_simulation.py for the measurement.
tt.set_arch("sequential")

tt.set_param("low_acc", False)

PERIODIC = ["periodic"] * 4

# Inflow at x=ax, outflow at x=bx, periodic side walls -- the bundle case.
BCV_CHANNEL = ["dirichlet", "neumann", "periodic", "periodic"]

BCP_CHANNEL = ["neumann", "dirichlet", "periodic", "periodic"]


class Operators:
    """Difference MPOs for one grid, built once and reused every step.

    `lap` is built as Dx@Dx + Dy@Dy from the *same* first-derivative
    operators used for the divergence and the gradient, so the projection
    identity div(grad phi) == lap(phi) holds by construction rather than by
    the coincidence that two separately-built stencils agree.
    """

    def __init__(self, Q, dx, dy=None, bcv=None, bcp=None, inflow_rows=4, held=False, inflow=None):
        dy = dx if dy is None else dy
        bcv = PERIODIC if bcv is None else bcv
        bcp = bcv if bcp is None else bcp
        self.Q, self.dx, self.dy = Q, dx, dy
        self.bcv, self.bcp = bcv, bcp
        nofq = [Q, Q]

        def d1(axis, bc, h):
            return tt.get_diff_mpo(dim=2, order=1, axis=axis, nofq=nofq, bc=bc) * (1.0 / h)

        def d2(axis, bc, h):
            return tt.get_diff_mpo(dim=2, order=2, axis=axis, nofq=nofq, bc=bc) * (1.0 / h**2)

        self.Dx, self.Dy = d1(0, bcv, dx), d1(1, bcv, dy)  # velocity
        self.Gx, self.Gy = d1(0, bcp, dx), d1(1, bcp, dy)  # pressure
        self.Dxx, self.Dyy = d2(0, bcv, dx), d2(1, bcv, dy)  # viscous
        # The Poisson operator is div(grad(.)) COMPOSED from the very
        # operators the projection uses, not a pair of second-difference
        # MPOs.  Built the latter way the two agree to 6e-11 in the interior
        # but by a factor of 55 at the inflow/outflow rows, and that
        # mismatch, not the solver, is what caps the projection: refining
        # the Poisson residual 100x (rank 20 -> 40) moved the resulting
        # divergence not at all, 1.194e-01 -> 1.192e-01.
        #
        # tensnet's MPO __matmul__ is REVERSED composition:
        # (A @ B).apply(x) == B.apply(A.apply(x)), verified to 8.5e-15.
        # So divergence-of-gradient is Gx @ Dx, not Dx @ Gx.
        self.lap = (self.Gx @ self.Dx) + (self.Gy @ self.Dy)

        # Boundary hold, BOTH ends.  Measured: d/dx of a constant, which is
        # exactly zero, comes back as O(40) in rows 0-3 and rows n-4..n-1
        # and as 1.5e-13 everywhere between.  So get_diff_mpo's stencil is
        # unusable in four rows at each end whatever `bc` says; tensnet
        # masks the inlet ones in construct_b and leaves the outlet ones in,
        # which is why the last column read +1.3 against -0.35 beside it.
        #
        # Zeroing an *increment* in those rows leaves them at whatever the
        # initial condition put there, which for this channel is u = U at
        # both ends -- mass-consistent, since inlet and outlet are equally
        # open and the obstacle is interior.  Without any hold at the inlet
        # a blocked domain simply drains: mask*U lost 95% of its norm in one
        # step and the mean ux read 0.0000 in every column.
        self.hold = None
        if inflow_rows and (bcv[0] != "periodic" or bcv[1] != "periodic"):
            idx = list(range(inflow_rows)) + list(range(2**Q - inflow_rows, 2**Q))
            rows = tt.row_mpo(Q, idx[0])
            for i in idx[1:]:
                rows = rows + tt.row_mpo(Q, i)
            keep = tt.identity_mpo(Q) + rows * tt.identity_mpo(Q) * -1.0
            self.hold = tt.get_2D_operator(keep, 0, nofq)

        # Held projection.  The unheld projection (held=False) holds the
        # predictor increment and the Poisson right-hand side in the
        # boundary rows but not the correction -grad(phi), and the "neumann"
        # rows of get_diff_mpo for phi at the inflow are not a Neumann
        # condition (rows 1-2 get a wrong stencil, rows 0 and 3 stay
        # homogeneous Dirichlet), so every projection with work near the
        # ends moves the inflow.  The
        # Star-CCM+ warm start is not mass-consistent here (outlet flux
        # 0.96 U) and its cleanup projection put the ENEC inflow at 0.959 U;
        # the inflow-boundary study in report/inlet.py has the measurements.  held=True holds the
        # correction too and solves the operator that implies,
        #   H div(H grad phi) + (I - H) phi = H div(u*),
        # i.e. phi = 0 in the held band, so the inflow never moves and the
        # interior divergence is removed as before.  The held rows carry
        # -(1/dx^2 + 1/dy^2) phi, the Laplacian's own sign and size (b = 0
        # there, so the solution is unchanged).  AMEn needs both: measured at
        # Q=8, chi=40 against the exact solve, the grad(phi) error after 6/12
        # sweeps is 0.14/0.012 this way and 0.13/0.010 for the default
        # operator; with +1 rows it stalls at 0.17, and with +(1/dx^2+1/dy^2)
        # it diverges (residual 4.4), the indefinite operator defeating the
        # Galerkin steps.  The outlet rows are then
        # touched by nothing, so each predictor stage gives them a
        # zero-gradient copy of the last unheld row (`outflow`).
        self.held_projection, self.outflow = held, None
        if held:
            if self.hold is None:
                raise ValueError("held=True needs held boundary rows (non-periodic x)")
            self.lap = (
                (self.Gx @ self.hold @ self.Dx @ self.hold)
                + (self.Gy @ self.Dy @ self.hold)
                + tt.get_2D_operator(rows * tt.identity_mpo(Q), 0, nofq)
                * -(1.0 / dx**2 + 1.0 / dy**2)
            )
            n, r = 2**Q, inflow_rows
            copy = np.eye(n)
            copy[n - r :] = 0.0
            copy[n - r :, n - r - 1] = 1.0
            copy = torch.as_tensor(copy, device=self.Dx.nodes[0].tensor.device)
            self.outflow = tt.get_2D_operator(tt.matrix_to_mpo(copy, 2, Q, 1e-14), 0, nofq)

        # Inflow pin (held only).  Compression is global, so every trim moves
        # the inflow rows a little, and with the correction held nothing ever
        # moves them back: at Q=6, chi=16 the inlet rms deviation still grew
        # to 4.5e-2 in 100 held steps.  `pin` resets them after each
        # projection, ux = U and uy = 0 there (the held correction never
        # touches those rows, so this only removes truncation error), and the
        # result is trimmed back to the cap: what is left in those rows is one
        # truncation's worth, not an accumulation (chi=16, Q=6: rms 3e-4 to
        # 9e-4 over 75 steps, mean 1.0000).
        self.pin = None
        if held and inflow is not None:
            dev = self.Dx.nodes[0].tensor.device
            keep_in = np.ones(2**Q)
            keep_in[:inflow_rows] = 0.0
            band = np.zeros((2**Q, 2**Q))
            band[:inflow_rows] = float(inflow)
            self.pin = (
                tt.get_2D_operator(
                    tt.matrix_to_mpo(torch.as_tensor(np.diag(keep_in), device=dev), 2, Q, 1e-14),
                    0,
                    nofq,
                ),
                grid_to_mps(band, Q).to(device=dev),
            )

    def held(self, m):
        """Zero an increment in the boundary rows, leaving the rest alone."""
        return m if self.hold is None else self.hold.apply(m)

    def divergence(self, ux, uy):
        """Plain Dx.ux + Dy.uy, with no hidden row surgery."""
        return self.Dx.apply(ux) + self.Dy.apply(uy)

    def laplacian(self, f):
        return self.Dxx.apply(f) + self.Dyy.apply(f)


POISSON_SOLVER = "amen"
"""Default Poisson solver.

NOT dmrg2. Verified against a dense reference with an exact solve on the
channel bundle's own right-hand side: dmrg2 stalls at relative residual
1.600 -- worse than returning zero -- and 200 sweeps give exactly the same
answer as 60, so it is stuck rather than slow. AMEn reaches 3.5e-13 in 20.

Two-site DMRG for Ax=b projects as (X^T A X) y = X^T b, a Galerkin
condition that only minimises anything when A is symmetric definite. The
channel Laplacian is not: ||A - A^T||/||A|| = 1.1e-01, because `lap`
composes first-derivative operators carrying *different* BC sets (Gx from
bcp, Dx from bcv). The periodic Laplacian IS symmetric to 2.4e-15, which is
why every periodic test here passed while the bundle never worked.

Symmetrising does not rescue it: on the normal equations A^T A, symmetric to
2.8e-16, dmrg2 still fails at residual 8.6e-01. So dmrg2 is inadequate for
this problem beyond the asymmetry, and AMEn is the solver to use.
"""


# ---- Time stepping -----------------------------------------------------------
def march_in_time(
    ux,
    uy,
    ops,
    dt,
    nu,
    Nt,
    mask=None,
    eps=1e-12,
    max_bd=None,
    opt_iter=6,
    uid=None,
    outdir="results/temp",
    local_solver=None,
    report=10,
    solver=POISSON_SOLVER,
    pressure=None,
    rotational=True,
    rhs_bd=None,
    guard=1.0,
    plain=False,
    rk4=False,
    phi_bd=None,
):
    """Advance Nt steps, writing one (ux, uy, p) snapshot per step.

    The per-step divergence is printed as |div u| * dx / |u|, dimensionless
    and O(1) when the projection has failed.  Every blow-up seen so far
    announced itself in that number long before the velocity misbehaved, so
    it is worth the one extra operator application per report.

    No outlet correction term.  tensnet's loop adds one to the right-hand
    side every step without resetting it, which grows |b| by 33% per step;
    here the pressure simply carries the homogeneous condition its BC set
    already implies, which is the standard p=0 outflow choice anyway.

    `plain` never feeds the pressure back into the predictor (plain Chorin,
    p = phi/dt each step); the pressure is still computed and saved.
    """
    import time

    solx, soly, solp = [], [], []
    guess = None
    for i in range(Nt):
        t0 = time.time()
        # A chunk's first solve starts cold (no previous phi).  At Q=11 six
        # AMEn sweeps from that start left a relative residual of 2.1-2.9
        # on the warm-started bundle, which the guard rightly rejects; twelve
        # reach 0.56.  Later steps start from the previous phi.
        ux, uy, pressure, guess = projection_step(
            ux,
            uy,
            ops,
            dt,
            nu,
            mask,
            eps,
            max_bd,
            opt_iter if guess is not None else max(opt_iter, 12),
            guess=guess,
            local_solver=local_solver,
            solver=solver,
            pressure=None if plain else pressure,
            rotational=rotational,
            rhs_bd=rhs_bd,
            guard=guard,
            rk4=rk4,
            phi_bd=phi_bd,
        )
        # kept and saved from the CPU: a pickled CUDA tensor does not load on
        # a machine without one, and a chunk's history would fill the GPU
        cx, cy, pr = (m.to(device="cpu") for m in (ux, uy, pressure))
        solx.append(cx)
        soly.append(cy)
        solp.append(pr)
        if uid is not None:
            os.makedirs(outdir, exist_ok=True)
            tt.save_object((cx, cy, pr), os.path.join(outdir, f"{uid}_int_sol_{i}.tmp"))
        if report and (i % report == 0 or i == Nt - 1):
            # Report the divergence over the rows the projection actually
            # targets.  The eight boundary rows are never corrected -- the
            # difference stencil is unusable there -- and including them
            # reads 5.3e-02 where the interior is 2.3e-05, which would make
            # every healthy run look broken.
            div = ops.divergence(ux, uy)
            scale = ops.dx / float(ux.norm)
            d = float(ops.held(div).norm) * scale
            res = getattr(project_divergence_free, "last_residual", float("nan"))
            print(
                f"  step {i:5d}  t={(i + 1) * dt:.4e}  |u|={float(ux.norm):.4e}"
                f"  |div|dx/|u|={d:.3e} (edges {float(div.norm) * scale:.1e})"
                f"  poisson={res:.1e}  rank={int(ux.max_rank)}"
                f"  {time.time() - t0:.1f}s",
                flush=True,
            )
            if not np.isfinite(d) or d > 0.5:
                raise RuntimeError(
                    f"projection failed at step {i}: |div| dx / |u| = {d:.3e}. "
                    f"The velocity is not divergence-free, so the run is "
                    f"meaningless from here. Lower the timestep or raise "
                    f"--opt-iter / --max-bd."
                )
    return solx, soly, solp


def projection_step(
    ux,
    uy,
    ops,
    dt,
    nu,
    mask=None,
    eps=1e-12,
    max_bd=None,
    opt_iter=6,
    guess=None,
    local_solver=None,
    solver=POISSON_SOLVER,
    pressure=None,
    rotational=True,
    rhs_bd=None,
    guard=1.0,
    rk4=False,
    phi_bd=None,
):
    """One step of rotational incremental pressure-correction.

    Plain Chorin sets p = phi/dt, so phi carries the entire pressure and its
    absolute error is amplified by 1/dt -- 2560x at dt=3.9e-4.  Measured on
    the warm-started bundle, that put the computed pressure 5x to 94x above
    the Star-CCM+ field with no trend over fourteen steps, while the
    velocity stayed healthy.  Carrying p instead makes phi the per-step
    *change*, which vanishes at a steady state.

    That also makes a steady solution an exact fixed point: if (u, p)
    balances the momentum equation then u* = u, so div(u*) = 0, phi = 0, and
    neither u nor p moves.  Starting from the reference and staying there is
    the whole validation.

    `rotational` adds the -nu*div(u*) term (Timmermans), which removes the
    O(dt) pressure boundary layer plain incremental leaves behind -- worth
    having because the control-volume force integrates pressure exactly
    where that error sits.
    """
    us, vs = predict_velocity(ux, uy, ops, dt, nu, mask, eps, max_bd, pressure, rk4)
    nx, ny, dp, phi = project_divergence_free(
        us,
        vs,
        ops,
        dt,
        eps,
        max_bd,
        opt_iter,
        max_rank=phi_bd,
        guess=guess,
        local_solver=local_solver,
        solver=solver,
        rhs_bd=rhs_bd,
        guard=guard,
    )
    with pinned_accuracy(eps):
        p_new = dp if pressure is None else pressure + dp
        if rotational:
            p_new = p_new + ops.held(ops.divergence(us, vs)) * (-nu)
        p_new = _compress_mps(p_new, eps, max_bd)
    return nx, ny, p_new, phi


def project_divergence_free(
    us,
    vs,
    ops,
    dt,
    eps=1e-12,
    max_bd=None,
    opt_iter=6,
    max_rank=None,
    guess=None,
    local_solver=None,
    solver=POISSON_SOLVER,
    rhs_bd=None,
    guard=1.0,
):
    """Remove the divergence of (us, vs).

    Solves lap(phi) = div(u*) and returns u* - grad(phi) together with the
    pressure phi/dt.  The right-hand side is computed here, from the field
    being corrected -- the defect in tensnet's loop is that it is not.
    """
    with pinned_accuracy(eps):
        # The inflow rows are not corrected, so their divergence must not be
        # put to phi either -- otherwise the solve chases an error the
        # correction is forbidden to fix.
        #
        # The right-hand side gets its OWN rank budget, wider than phi's.
        # Under the single global cap it was truncated to max_bd along with
        # everything else, which on the warm-started bundle discarded 25% of
        # it at max_bd=40 (11.7% at 80, 0.5% at 160): the solve then
        # converges neatly onto a problem that is missing a quarter of its
        # data, and reports a small residual while the divergence survives.
        # b enters the DMRG environments linearly, so widening it is cheap --
        # measured 5% more time for 22% less divergence.
        with compression_limits(eps, rhs_bd):
            b = ops.held(ops.divergence(us, vs))
        x0 = guess if guess is not None else tt.identity_mps(2 * ops.Q) * dt
        # `max_rank` lets phi carry more rank than the fields.  At chi=40 the
        # best possible rank-40 phi (exact phi, TT-SVD truncated) still has a
        # 53-56% error in grad(phi) on the bundle, 18-21% at 80, 6-7% at 120
        # (poisson_rank_benchmark.py), and every solver variant sits at that floor.
        # The global BD_LIMIT is lifted to match for the solve.
        with compression_limits(eps, max_rank or "keep"):
            phi = _solve_poisson(
                ops.lap, b, x0, eps, max_rank or max_bd or 40, opt_iter, solver, local_solver
            )
        # Verify the solve. None of the projection failures chased through
        # this module were visible because nothing ever checked: dmrg2 was
        # returning a phi with relative residual 1.600 -- worse than zero --
        # and reporting nothing. A residual near or above 1 means the solver
        # did not solve; treat it as an error rather than integrating it.
        # `guard` is the fatal threshold: 0.5 killed healthy runs on routine
        # excursions (0.594), so the default only catches phi worse than 0.
        #
        # A divergence at roundoff (a uniform stream) has no meaningful
        # relative residual -- measured 0.66 to 6e4 run to run -- so it is
        # only checked when b is non-negligible against |u|/dx.
        nb = float(b.norm)
        resid = float((ops.lap.apply(phi) + b * (-1.0)).norm) / nb if nb else 0.0
        if resid >= guard and nb > 1e-8 * float(us.norm) / ops.dx:
            raise RuntimeError(
                f"Poisson solve failed: relative residual {resid:.3e} "
                f"(solver={solver}, max_rank={max_rank or max_bd}, "
                f"sweeps={opt_iter}). Residual >= 1 means the answer is worse "
                f"than phi=0. dmrg2 does this on non-symmetric operators; "
                f"use solver='amen'."
            )
        project_divergence_free.last_residual = resid
        # The correction is NOT held.  Holding the right-hand side is what
        # restricts the projection to the rows we intend to fix; holding the
        # correction as well breaks the projection identity, because `lap` is
        # div(grad(.)) with no hold inside it.  Measured against a dense
        # reference with an exact solve (dense_projection_check.py), starting from
        # |held(div)| = 1.560e+02:
        #
        #   no hold anywhere      -> 1.8e-12     hold RHS only   -> 4.2e-12
        #   hold correction only  -> 5.4e+01     hold both       -> 2.7e+01
        #
        # So holding both, which this did, left a fifth of the divergence in
        # place even with an exact Poisson solve. The predictor increment is
        # still held, so the boundary rows keep their inflow value up to the
        # projection's own correction there.  That is the default; with
        # ops.held_projection the correction IS held, and `lap` is the
        # matching div(H grad(.)) operator, so the identity holds again.
        # u* - grad(phi) formed at phi's rank, THEN trimmed to the fields' cap:
        # trimming grad(phi) first would throw away what the extra rank bought.
        with compression_limits(eps, None if max_rank else "keep"):
            gx, gy = ops.Gx.apply(phi), ops.Gy.apply(phi)
            if ops.held_projection:
                gx, gy = ops.held(gx), ops.held(gy)
            ux = _compress_mps(us + gx * (-1.0), eps, max_bd)
            uy = _compress_mps(vs + gy * (-1.0), eps, max_bd)
        if (
            ops.pin is not None
        ):  # cap lifted: bundle_simulation sets a global BD_LIMIT, which would re-truncate
            # Then back to the fields' cap: the pinned sum carries the inflow
            # rows' truncation noise twice over and doubles the rank (Q=11:
            # 40 -> 81, the next products cost accordingly).  Trimming costs
            # one truncation's worth of inflow error, which the next pin
            # removes again, so it stays at that size instead of accumulating.
            with compression_limits(eps, None):
                ux = ops.pin[0].apply(ux) + ops.pin[1]
                uy = ops.pin[0].apply(uy)
            ux, uy = _compress_mps(ux, eps, max_bd), _compress_mps(uy, eps, max_bd)
        return ux, uy, phi * (1.0 / dt), phi


def predict_velocity(
    ux, uy, ops, dt, nu, mask=None, eps=1e-12, max_bd=None, pressure=None, rk4=False
):
    """Explicit advection + diffusion, then the mask.

    Masking here rather than after the projection is the whole point: the
    projection that follows removes the divergence the mask introduces.

    `rk4` takes a classical RK4 step (pressure gradient frozen over the step,
    every stage masked and trimmed) instead of forward Euler.  Euler with
    these eighth-order central differences is linearly unstable on the
    bundle: at CFL 0.4 / cell Re 6.25 the real Q=10 operators, boundary rows
    included, give |G| = 1.11 per step at U and 1.39 at 1.33U, and the exact-
    arithmetic twin (uncompressed_twin.py) blows up after ~45 steps.  The MPS runs stayed
    bounded only because truncation damped the growing modes.  RK4 gives
    |G| <= 1 to 2U at the same dt, for four right-hand-side evaluations.
    """
    with pinned_accuracy(eps):

        def rhs(u, v):
            ix = (u * ops.Dx.apply(u) + v * ops.Dy.apply(u)) * (-1.0) + ops.laplacian(u) * nu
            iy = (u * ops.Dx.apply(v) + v * ops.Dy.apply(v)) * (-1.0) + ops.laplacian(v) * nu
            if pressure is not None:
                # The incremental term.  With it, phi is only the *change* in
                # pressure over a step rather than the whole of it, which is what
                # keeps p = phi/dt from amplifying the solve error by 1/dt.
                ix = ix + ops.Gx.apply(pressure) * (-1.0)
                iy = iy + ops.Gy.apply(pressure) * (-1.0)
            return ops.held(ix), ops.held(iy)

        def stage(c, k):
            u, v = ux + k[0] * (c * dt), uy + k[1] * (c * dt)
            if mask is not None:
                u, v = u * mask, v * mask
            if ops.outflow is not None:
                u, v = ops.outflow.apply(u), ops.outflow.apply(v)
            return _compress_mps(u, eps, max_bd), _compress_mps(v, eps, max_bd)

        k = rhs(ux, uy)
        if rk4:
            k2 = rhs(*stage(0.5, k))
            k3 = rhs(*stage(0.5, k2))
            k4 = rhs(*stage(1.0, k3))
            k = tuple(
                (a + b * 2.0 + c * 2.0 + d) * (1.0 / 6.0) for a, b, c, d in zip(k, k2, k3, k4)
            )
        return stage(1.0, k)


def _solve_poisson(A, b, x0, eps, max_rank, opt_iter, solver, local_solver):
    """Solve A phi = b with whichever tensnet routine was asked for.

    Measured on this operator (Q=8, rank 40, 6 sweeps): dmrg2 with its
    default dense supercore solve takes 72s for a 1.6e-03 residual; AMEn
    takes 4.9s for 7.0e-02 and 65s for 2.1e-03, so it is a genuinely
    cheaper operating point only if a loose solve per step suffices.
    Every iterative local solver -- cg, gmres, matfree -- was slower than
    the direct one for identical accuracy, so `local_solver` is offered but
    not recommended.
    """
    if solver == "amen":
        sol = tt.linsys_amen(A=A, b=b, x=x0, max_rank=max_rank, max_iter=opt_iter, eps=eps)
        return sol[0] * torch.exp(sol[1]) if isinstance(sol, tuple) else sol
    sol = tt.linsys_dmrg2(
        A=A,
        b=b,
        x=x0,
        trunc_eps=eps,
        max_iter=opt_iter,
        max_rank=max_rank,
        local_solver=local_solver,
        verbose=False,
    )
    return sol[0] * torch.exp(sol[1])


# ---- Compression settings ----------------------------------------------------
@contextlib.contextmanager
def compression_limits(eps, bd="keep"):
    """Pin tensnet's global compression accuracy and bond-dimension cap."""
    saved_a, saved_b = tt.COMP_ACC(), tt.BD_LIMIT()
    tt.set_accuracy(eps)
    if bd != "keep":
        tt.set_bd_limit(bd)
    try:
        yield
    finally:
        tt.set_accuracy(saved_a)
        tt.set_bd_limit(saved_b)


@contextlib.contextmanager
def pinned_accuracy(eps):
    """Pin tensnet's global compression accuracy for a block of work.

    MPS arithmetic compresses against the global COMP_ACC, not against the
    accuracy handed to compress(), so every routine here that adds or
    multiplies MPS has to set it.  Without this the module silently inherits
    whatever the caller left behind -- at the 1e-6 default the projection
    leaves 4e-02 of the divergence it removes completely at 1e-12, which
    reads as a broken projection rather than a loose tolerance.
    """
    saved = tt.COMP_ACC()
    tt.set_accuracy(eps)
    try:
        yield
    finally:
        tt.set_accuracy(saved)


def _compress_mps(m, eps, max_bd):
    return m.compress(max_bd=max_bd, accuracy=eps) if max_bd else m.compress(accuracy=eps)


# ---- Encode and decode -------------------------------------------------------
def grid_to_mps(field, Q, eps=1e-12):
    return tt.vector_to_mps(tt.shuffle_axis(torch.as_tensor(field), Q).flatten(), 2, 2 * Q, eps)


def mps_to_grid(m, Q):
    return tt.unshuffle_axis(m.contract(), Q, 2).numpy()


# ---- Test suite (python3 chorin_projection.py) -------------------------------
def selfcheck():
    tests = [
        test_projection_identity,
        test_projection_removes_divergence,
        test_divergence_free_is_preserved,
        test_mask_before_projection,
        test_boundary_rows_are_clean,
        test_inflow_is_held,
        test_held_projection_keeps_inflow,
        test_steady_state_is_a_fixed_point,
        test_taylor_green,
    ]
    import time

    bad = 0
    for fn in tests:
        t0 = time.time()
        try:
            msg = fn()
            print(f"  PASS  [{time.time() - t0:5.1f}s] {msg}", flush=True)
        except AssertionError as exc:
            print(f"  FAIL  [{time.time() - t0:5.1f}s] {fn.__name__}: {exc}", flush=True)
            bad += 1
    print("chorin_projection: OK" if not bad else f"chorin_projection: {bad} FAILED")
    return bad


def test_projection_identity(Q=7):
    """lap must be exactly div(grad(.)) or the projection cannot work.

    Whole domain, boundaries included.  Built from second-difference MPOs
    instead, this held to 6e-11 in the interior but was off by a factor of
    55 at the inflow/outflow rows, and those rows dominated: refining the
    Poisson residual 100x left the divergence unchanged.  Composing the
    operator out of the ones the projection actually applies makes the
    identity exact everywhere, which is the whole point of the test.
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx, bcv=BCV_CHANNEL, bcp=BCP_CHANNEL)
    phi = grid_to_mps(np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y), Q)
    lhs = mps_to_grid(ops.lap.apply(phi), Q)
    rhs = mps_to_grid(ops.Dx.apply(ops.Gx.apply(phi)) + ops.Dy.apply(ops.Gy.apply(phi)), Q)
    rel = np.abs(lhs - rhs).max() / np.abs(lhs).max()
    assert rel < 1e-10, f"lap != div.grad, relative {rel:.2e}"
    return f"projection identity holds over the whole domain ({rel:.1e})"


def test_projection_removes_divergence(Q=6):
    """A pure gradient field must project to nothing.

    u = grad(chi) is all divergence and no solenoidal part, so the exact
    answer is zero.  It also keeps the periodic Poisson problem solvable:
    div(grad(chi)) has zero mean by construction, whereas a field with a
    mean divergence makes the periodic Laplacian singular -- which is what
    an earlier version of this test hit, and why it reported that the
    projection "left 99.7% of the divergence".
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx)
    chi = grid_to_mps(np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y), Q)
    us, vs = ops.Gx.apply(chi), ops.Gy.apply(chi)
    n0 = float(us.norm)
    ux, uy, _, _ = project_divergence_free(
        us, vs, ops, dt=1e-3, opt_iter=12, max_rank=32, max_bd=32
    )
    left = float(ux.norm) / n0
    assert left < 0.05, f"projection left {100 * left:.1f}% of a pure gradient"
    return f"pure gradient projects away ({100 * left:.3f}% left)"


def test_divergence_free_is_preserved(Q=6):
    """Projecting an already divergence-free field must not disturb it."""
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx)
    psi = grid_to_mps(np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y), Q)
    ux, uy = ops.Dy.apply(psi), ops.Dx.apply(psi) * (-1.0)
    n0 = float(ux.norm)
    assert float(ops.divergence(ux, uy).norm) / (n0 / dx) < 1e-9
    px, py, _, _ = project_divergence_free(ux, uy, ops, dt=1e-3, opt_iter=8, max_rank=32, max_bd=32)
    rel = float((px + ux * (-1.0)).norm) / n0
    assert rel < 1e-3, f"projection moved a divergence-free field by {rel:.2e}"
    return f"divergence-free field survives projection (moved {rel:.1e})"


def test_mask_before_projection(Q=6):
    """The ordering that tensnet gets wrong, stated as a test.

    Masking after projecting reintroduces the divergence; masking before it
    does not, because the projection then cleans up after the mask.
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx)
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    mask = grid_to_mps(1.0 / (1.0 + np.exp(-(r - 0.15) / (2 * dx))), Q)
    psi = grid_to_mps(np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y), Q)
    ux, uy = ops.Dy.apply(psi), ops.Dx.apply(psi) * (-1.0)
    scale = float(ux.norm) / dx

    after = float(ops.divergence(ux * mask, uy * mask).norm) / scale
    bx, by, _, _ = project_divergence_free(
        ux * mask, uy * mask, ops, dt=1e-3, opt_iter=8, max_rank=32, max_bd=32
    )
    before = float(ops.divergence(bx, by).norm) / scale
    assert before < after / 10, (
        f"mask-then-project ({before:.2e}) is not better than project-then-mask ({after:.2e})"
    )
    return (
        f"mask before projection: {before:.2e} vs {after:.2e} after ({after / before:.0f}x worse)"
    )


def test_boundary_rows_are_clean(Q=6, nrows=4):
    """State the boundary defect, and that `held` removes it.

    d/dx of a constant is exactly zero.  get_diff_mpo returns O(40) for it
    in `nrows` rows at each end whatever the bc list says -- the condition
    lives in the stencil, but the stencil is not usable that close to the
    edge.  tensnet masks the inlet rows in construct_b and leaves the
    outlet ones, which is what put +1.3 in the last column against -0.35
    beside it.
    """
    (X, _), dx = _unit_square_grid(Q)
    n = 2**Q
    ops = Operators(Q, dx, bcv=BCV_CHANNEL, bcp=BCP_CHANNEL, inflow_rows=nrows)
    one = grid_to_mps(np.ones_like(X), Q)
    edge = list(range(nrows)) + list(range(n - nrows, n))
    for lab, op in (("Dx", ops.Dx), ("Gx", ops.Gx)):
        raw = mps_to_grid(op.apply(one), Q)[:, n // 2]
        interior = np.delete(raw, edge)
        assert np.abs(interior).max() < 1e-9, (
            f"{lab} does not annihilate a constant in the interior: {np.abs(interior).max():.2e}"
        )
        assert np.abs(raw[edge]).max() > 1.0, (
            f"{lab}'s boundary rows look clean ({np.abs(raw[edge]).max():.2e}); "
            f"if that is now true the hold may be unnecessary"
        )
        clean = np.abs(mps_to_grid(ops.held(op.apply(one)), Q)).max()
        assert clean < 1e-9, f"held({lab}(const)) = {clean:.2e}, not zero"
    return (
        f"boundary rows isolated: {nrows} at each end, interior clean to 1e-9, held() zeroes them"
    )


def test_inflow_is_held(Q=6, steps=3):
    """Flow past an obstacle must not drain out of the domain.

    tensnet's difference operators carry a *homogeneous* Dirichlet
    condition, so nothing in them says u = U at the inlet.  Without the
    inflow hold, mask*U lost 95% of its norm in a single step and the mean
    ux read 0.0000 in every column -- inlet, near field and far field
    alike.  The projection was blameless: a bare uniform flow survives it
    intact (100% periodic, 94% channel).  It was the blocked domain with
    nothing driving it.
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx, bcv=BCV_CHANNEL, bcp=BCP_CHANNEL)
    r = np.sqrt((X - 0.35) ** 2 + (Y - 0.5) ** 2)
    mask = grid_to_mps(1.0 / (1.0 + np.exp(-(r - 0.12) / (2 * dx))), Q)
    U = 1.0
    ux, uy = mask * U, mask * 0.0
    n0, inlet0 = float(ux.norm), _column_mean(ux, 0, Q)
    guess = None
    for _ in range(steps):
        ux, uy, _, guess = projection_step(
            ux, uy, ops, dx * 0.2 / U, 1e-3, mask, eps=1e-11, max_bd=32, opt_iter=8, guess=guess
        )
    inlet, far = _column_mean(ux, 0, Q), _column_mean(ux, 2**Q - 8, Q)
    kept = float(ux.norm) / n0
    assert kept > 0.5, f"the flow drained: |u| kept only {100 * kept:.1f}%"
    assert abs(inlet - inlet0) < 0.05 * U, f"inlet drifted from {inlet0:.4f} to {inlet:.4f}"
    assert far > 0.3 * U, f"far field collapsed to {far:.4f}"
    return (
        f"inflow held over {steps} steps (inlet {inlet:.4f}, far field "
        f"{far:.4f}, |u| kept {100 * kept:.0f}%)"
    )


def test_held_projection_keeps_inflow(Q=6, steps=3):
    """The held projection must leave the inflow rows alone; the default does not.

    The field is the ENEC situation in miniature: flow past a disc whose
    outlet band carries 4 % less flux than the inlet, as the Star-CCM+ warm
    start does in this domain (0.96 U).  Projecting it with the default
    operators moves the inflow; with held=True the inlet rows must come back
    exactly, the interior divergence must still be removed, and a few steps
    later the inflow must still be U.
    """
    (X, Y), dx = _unit_square_grid(Q)
    n = 2**Q
    r = np.sqrt((X - 0.35) ** 2 + (Y - 0.5) ** 2)
    mask_g = 1.0 / (1.0 + np.exp(-(r - 0.12) / (0.5 * dx)))  # 1 at the ends
    ramp = np.clip((np.arange(n) - (n - 12)) / 8.0, 0, 1)[:, None]
    mask, U = grid_to_mps(mask_g, Q), 1.0
    inlet = {}
    for held in (False, True):
        ops = Operators(Q, dx, bcv=BCV_CHANNEL, bcp=BCP_CHANNEL, held=held)
        ux = grid_to_mps(mask_g * U * (1 - 0.04 * ramp), Q)
        uy = grid_to_mps(np.zeros_like(X), Q)
        b0 = float(ops.held(ops.divergence(ux, uy)).norm)
        ux, uy, _, _ = project_divergence_free(
            ux, uy, ops, 0.2 * dx / U, eps=1e-12, max_bd=64, opt_iter=20
        )
        g = mps_to_grid(ux, Q)
        inlet[held] = np.abs(g[:4] - U).max()
        div = float(ops.held(ops.divergence(ux, uy)).norm) / b0
        assert div < 1e-6, f"held={held}: projection left {div:.1e} of the divergence"
    assert inlet[False] > 1e-3, (
        f"default projection no longer moves the inflow ({inlet[False]:.1e})"
    )
    assert inlet[True] < 1e-9, f"held projection moved the inflow by {inlet[True]:.1e}"
    # now under truncation (chi=12, also as the global cap, as bundle_simulation sets
    # it): the pin must keep the inflow exact anyway
    ops = Operators(Q, dx, bcv=BCV_CHANNEL, bcp=BCP_CHANNEL, held=True, inflow=U)
    guess, pr = None, grid_to_mps(np.zeros_like(X), Q)
    with compression_limits(1e-12, 12):
        for _ in range(steps):
            ux, uy, pr, guess = projection_step(
                ux,
                uy,
                ops,
                0.2 * dx / U,
                1e-3,
                mask,
                eps=1e-12,
                max_bd=12,
                opt_iter=12,
                guess=guess,
                pressure=pr,
            )
    g = mps_to_grid(ux, Q)[:4]
    moved, rms = abs(g.mean() - U), np.sqrt(((g - U) ** 2).mean())
    # one truncation's worth, not accumulated (unpinned: rms 4.5e-2 by step 100 at chi=16)
    assert moved < 1e-3 and rms < 1e-2, (
        f"pinned inflow off by {moved:.1e} (mean), {rms:.1e} (rms) after {steps} steps at chi=12"
    )
    return (
        f"inflow moved {inlet[False]:.1e} by the default projection, "
        f"{inlet[True]:.0e} held; after {steps} pinned steps at chi=12 "
        f"mean {moved:.0e}, rms {rms:.0e}"
    )


def test_steady_state_is_a_fixed_point(Q=6, steps=5):
    """A steady solution, fed in with its pressure, must not move.

    On an unforced periodic box the only steady solution is a uniform
    stream with constant pressure -- everything else decays viscously, so
    an earlier version of this test that used a Taylor-Green field was
    asserting a property the field does not have.  Trivial as it is, this
    catches a scheme that manufactures pressure or velocity out of nothing,
    which is exactly what plain Chorin does through phi/dt.

    The non-trivial pressure coupling is covered by test_taylor_green,
    which seeds the exact unsteady pressure and checks the decay rate.
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx)
    U = 1.3
    ux = grid_to_mps(np.full_like(X, U), Q)
    uy = grid_to_mps(np.zeros_like(X), Q)
    p = grid_to_mps(np.full_like(X, 0.7), Q)
    u0 = float(ux.norm)
    a_, b_, pp, g = ux, uy, p, None
    for _ in range(steps):
        a_, b_, pp, g = projection_step(
            a_, b_, ops, 0.2 * dx / U, 5e-3, eps=1e-11, max_bd=32, opt_iter=10, guess=g, pressure=pp
        )
    drift = float((a_ + ux * (-1.0)).norm) / u0
    cross = float(b_.norm) / u0
    assert drift < 1e-3, f"uniform stream drifted {100 * drift:.3f}%"
    assert cross < 1e-3, f"cross-stream velocity appeared: {cross:.2e}"
    return (
        f"uniform steady state held for {steps} steps (drift {drift:.1e}, cross flow {cross:.1e})"
    )


def test_taylor_green(Q=6, steps=12):
    """Decay rate of the Taylor-Green vortex, an exact solution of NS.

    u = -cos(2 pi x) sin(2 pi y) e^(-8 pi^2 nu t), and the nonlinear term is
    balanced by the pressure gradient, so this exercises advection, diffusion
    and the projection together rather than any one of them.
    """
    (X, Y), dx = _unit_square_grid(Q)
    ops = Operators(Q, dx)
    nu = 1e-2
    # BOTH limits, not just the diffusive one.  An earlier version of this
    # test took dt from the diffusive limit alone, which put the advective
    # CFL at 1.59 and made Taylor-Green grow by 1e8 -- a test bug that looks
    # exactly like a broken solver.
    dt = min(0.4 * dx / 1.0, 0.2 * dx**2 / nu)
    u0 = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    v0 = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    # The exact Taylor-Green pressure, so the incremental path carries a
    # real field rather than starting from nothing.
    p0 = -0.25 * (np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y))
    ux, uy, pr = grid_to_mps(u0, Q), grid_to_mps(v0, Q), grid_to_mps(p0, Q)
    n0 = float(ux.norm)
    guess = None
    for _ in range(steps):
        ux, uy, pr, guess = projection_step(
            ux, uy, ops, dt, nu, eps=1e-10, max_bd=16, opt_iter=6, guess=guess, pressure=pr
        )
    t = steps * dt
    want = np.exp(-8 * np.pi**2 * nu * t)
    got = float(ux.norm) / n0
    rel = abs(got - want) / want
    div = float(ops.divergence(ux, uy).norm) / (float(ux.norm) / dx)
    # 2%: the scheme's own error here is ~1e-3, and a solver that simply
    # failed to decay at all would come in at ~4.8%, so this discriminates.
    assert rel < 0.02, f"Taylor-Green decayed by {got:.5f}, exact {want:.5f}"
    assert div < 1e-3, f"Taylor-Green lost divergence-freedom ({div:.2e})"
    return (
        f"Taylor-Green decay {got:.5f} vs exact {want:.5f} ({100 * rel:.2f}% off, |div| {div:.1e})"
    )


def _unit_square_grid(Q):
    n = 2**Q
    g = np.linspace(0.0, 1.0, n)
    return np.meshgrid(g, g, indexing="ij"), 1.0 / (n - 1)


def _column_mean(m, ix, Q):
    """Mean of an MPS field down one column of constant x."""
    n = 2**Q
    flat = np.zeros(n, dtype=np.int64)
    for k in range(Q):
        flat = (flat << 1) | ((ix >> (Q - 1 - k)) & 1)
        flat = (flat << 1) | ((np.arange(n) >> (Q - 1 - k)) & 1)
    return float(np.asarray(m.query(flat.tolist())).mean())


if __name__ == "__main__":
    raise SystemExit(selfcheck())
