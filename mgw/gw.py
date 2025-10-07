import numpy as np
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
from ot.gromov import gwloss, init_matrix  # POT, for diagnostics only
import jax
import jax.numpy as jnp
from ott.geometry import geometry

def gw_distance(C1, C2, p, q):
    """
    Compute the (squared) GW objective value given cost matrices and coupling.
    Useful for monitoring, not for optimization itself.
    """
    # GW objective: sum_{i,i',j,j'} (C1_{ii'} - C2_{jj'})^2 * P_{ij} P_{i'j'}
    term = (C1[:, :, None, None] - C2[None, None, :, :]) ** 2
    val = np.sum(term * (p[:, None, :, None] * p[None, :, None, :]))  # if p==P (not just marginals)
    return val

def solve_gw(C1, C2, a=None, b=None,
             epsilon=1e-2,
             numItermax=500,
             tol=1e-6,
             verbose=True,
             use_ott=True,
             normalize=False):
    """
    Solve entropic Gromov–Wasserstein using OTT-JAX (preferred) or POT fallback.
    Compatible with recent OTT-JAX API (requires linear_solver).
    """
    n1, n2 = C1.shape[0], C2.shape[0]
    if a is None: a = np.ones(n1) / n1
    if b is None: b = np.ones(n2) / n2

    # normalize costs to comparable scale
    if normalize:
        q99 = np.quantile(np.concatenate([C1.ravel(), C2.ravel()]), 0.99)
        if q99 > 0:
            C1 = C1 / q99
            C2 = C2 / q99

    if use_ott:
        try:
            from ott.problems.quadratic import quadratic_problem
            from ott.solvers.linear.sinkhorn import Sinkhorn
            from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
        except ImportError:
            use_ott = False
            if verbose:
                print("[solve_gw] Falling back to POT (OTT-JAX not installed).")

    if use_ott:
        # convert to JAX
        C1_j = jnp.asarray(C1, dtype=jnp.float32)
        C2_j = jnp.asarray(C2, dtype=jnp.float32)
        a_j  = jnp.asarray(a, dtype=jnp.float32)
        b_j  = jnp.asarray(b, dtype=jnp.float32)

        # geometries for the two cost matrices
        geom_x = geometry.Geometry(cost_matrix=C1_j, epsilon=None)
        geom_y = geometry.Geometry(cost_matrix=C2_j, epsilon=None)

        # quadratic problem
        prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a_j, b=b_j)

        # linear solver (Sinkhorn) required in OTT ≥0.5
        linear_solver = Sinkhorn(max_iterations=numItermax, threshold=tol)

        # Gromov–Wasserstein solver
        solver = GromovWasserstein(
            linear_solver=linear_solver,
            epsilon=epsilon,
            max_iterations=numItermax,
            threshold=tol
        )

        # solve
        out = solver(prob)
        P = np.array(out.matrix)
        if verbose:
            print(f"[OTT] GW converged: loss={float(out.primal_cost):.4e}, "
                  f"ε={epsilon}, iters={out.n_iters}")
        return P

    # ---- fallback: POT ----
    import ot
    if epsilon > 0:
        P = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, a, b, loss_fun='square_loss',
            epsilon=epsilon, verbose=verbose, max_iter=numItermax)
    else:
        P = ot.gromov.gromov_wasserstein(
            C1, C2, a, b, loss_fun='square_loss',
            verbose=verbose, max_iter=numItermax)
    return P

def solve_gw_ott(C1, C2, a=None, b=None,
                        epsilon=1e-2, inner_tol=1e-7, outer_tol=1e-7,
                        inner_maxit=4000, outer_maxit=1000, verbose=True, jit=True):
    
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    
    if a is None: a = np.ones(n1)/n1
    if b is None: b = np.ones(n2)/n2
    
    # JAX arrays
    C1_j = jnp.asarray(C1, dtype=jnp.float32)
    C2_j = jnp.asarray(C2, dtype=jnp.float32)
    a_j  = jnp.asarray(a,  dtype=jnp.float32)
    b_j  = jnp.asarray(b,  dtype=jnp.float32)
    
    # Intra-domain geometries
    geom_x = geometry.Geometry(cost_matrix=C1_j, epsilon=epsilon)
    geom_y = geometry.Geometry(cost_matrix=C2_j, epsilon=epsilon)
    
    # Quadratic GW problem
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a_j, b=b_j)
    
    # Inner linear solver + outer GW solver
    lin = Sinkhorn(max_iterations=inner_maxit, threshold=inner_tol)
    gw  = GromovWasserstein(linear_solver=lin, epsilon=epsilon,
                            max_iterations=outer_maxit, threshold=outer_tol)
    
    if jit:
        gw = jax.jit(gw)
    
    out = gw(prob)
    P = np.array(out.matrix)
    
    # Diagnostics (POT’s explicit GW loss; optional)
    constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun="square_loss")
    loss_val = gwloss(constC, hC1, hC2, P)
    
    if verbose:
        print(f"[OTT] GW  ε={epsilon:.2e}  iters={getattr(out,'n_iters',-1)}  loss≈{loss_val:.4e}")
    return P

