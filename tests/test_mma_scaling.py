"""MMA scaling: the optimizer must be invariant to the physical units of objective and
constraints because it normalizes both itself (objective by |F(x0)|, rows by |target|)."""

import numpy as np
import torch

from DeepSDFStruct.optimization import MMA


def _run(obj_scale, con_scale, n_steps=15, pass_scale=True):
    """min obj_scale * ||x - c||^2  s.t.  con_scale * (sum(x) - budget) <= 0."""
    torch.manual_seed(0)
    c = torch.tensor([1.0, 2.0, -0.5, 0.8], dtype=torch.float64)
    budget = 1.5
    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    bounds = np.array([[-3.0, 3.0]] * 4)
    opt = MMA(x, bounds, max_step=0.3, n_constraints=1)
    for _ in range(n_steps):
        F = obj_scale * ((x - c) ** 2).sum()
        value = con_scale * x.sum()
        target = con_scale * budget
        G = value - target
        dF = torch.autograd.grad(F, x, retain_graph=True)[0]
        dG = torch.autograd.grad(G, x)[0]
        opt.step(F.detach(), dF, G.detach().reshape(1), dG.reshape(1, -1),
                 G_scale=[target] if pass_scale else None)
    return x.detach().numpy().copy(), opt


def test_scaling_invariance():
    x_ref, opt_ref = _run(1.0, 1.0)
    x_small, opt_small = _run(1e-6, 1e-5)   # CFD-like magnitudes
    x_big, _ = _run(1e4, 3e2)
    assert np.allclose(x_ref, x_small, atol=1e-8), (x_ref, x_small)
    assert np.allclose(x_ref, x_big, atol=1e-8), (x_ref, x_big)
    assert opt_ref.F0 > 0 and opt_small.F0 > 0
    assert np.isclose(opt_small.G_scale[0, 0], 1e-5 * 1.5)
    # The constraint is active at the optimum: sum(c) = 3.3 > budget 1.5.
    assert x_ref.sum() <= 1.5 + 1e-6


def test_unscaled_rows_fall_back_to_one():
    _, opt = _run(1.0, 1.0, n_steps=2, pass_scale=False)
    assert np.all(opt.G_scale == 1.0)
    _, opt = _run(1.0, 1.0, n_steps=2)
    assert opt._row_scales([None])[0, 0] == 1.0
    assert opt._row_scales([0.0])[0, 0] == 1.0
    assert opt._row_scales([-2.0])[0, 0] == 2.0


def test_ch_is_mean_abs_change():
    _, opt = _run(1.0, 1.0, n_steps=3)
    expected = np.abs(opt.x - opt.xold1).mean() / np.abs(opt.x).mean()
    assert np.isclose(opt.ch, expected) and opt.ch > 0


def test_negative_initial_objective_keeps_descent():
    # F = -(x^2) shifted negative: normalizing by a SIGNED F0 would flip the direction.
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt = MMA(x, np.array([[-5.0, 5.0]]), max_step=0.5, n_constraints=1)
    for _ in range(3):
        F = (x - 3.0) ** 2 - 100.0           # negative, minimum at x = 3
        dF = torch.autograd.grad(F.sum(), x)[0]
        G = torch.zeros(1, dtype=torch.float64) - 1.0   # inactive row
        opt.step(F.detach().reshape(1), dF, G, torch.zeros(1, 1, dtype=torch.float64))
    assert x.item() > 1.0, x.item()


def test_kkt_residual_vanishes_at_constrained_optimum():
    _, opt_first = _run(1.0, 1.0, n_steps=1)
    _, opt_late = _run(1.0, 1.0, n_steps=40)
    _, opt_late_small = _run(1e-6, 1e-5, n_steps=40)
    assert np.isfinite(opt_first.kkt_norm) and opt_first.kkt_norm > 0
    # The constraint is active at the optimum, so the objective gradient alone does NOT
    # vanish there -- only the Lagrangian gradient does.
    assert opt_late.kkt_norm < 1e-2 * opt_first.kkt_norm, (opt_first.kkt_norm, opt_late.kkt_norm)
    assert opt_late.lam[0, 0] > 0
    # Scaled units: the residual does not depend on the physical magnitudes.
    assert np.isclose(opt_late.kkt_norm, opt_late_small.kkt_norm, rtol=1e-6, atol=1e-12)


def _kkt_single_variable(x0):
    """min 2 - x on [-1, 1]: descent pushes x out through the upper bound."""
    x = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
    opt = MMA(x, np.array([[-1.0, 1.0]]), max_step=0.1, n_constraints=1)
    F = 2.0 - x
    dF = torch.autograd.grad(F.sum(), x)[0]
    opt.step(F.detach().reshape(1), dF, torch.full((1,), -1.0, dtype=torch.float64),
             torch.zeros(1, 1, dtype=torch.float64))
    return opt


def test_kkt_residual_projects_onto_true_bounds():
    # At the upper bound the outward descent is blocked: stationary, residual 0.
    assert _kkt_single_variable(1.0).kkt_norm == 0.0
    # Interior point: full scaled gradient |dF/F0| = 1/2 (F0 = |F(0)| = 2), although the
    # step itself is capped by the move limit.
    assert np.isclose(_kkt_single_variable(0.0).kkt_norm, 0.5)
