from functools import lru_cache
from typing import Callable
import torch
from tqdm import tqdm

from ..linalg import get_diff_mpo
from ..networks import MPS, MPO, random_mps
from ..utils import parameters


class fluid_state:
    def __init__(self, ux: MPS, uy: MPS, rho: MPS, E: MPS):
        self.ux = ux
        self.uy = uy
        self.rho = rho
        self.E = E

    @property
    @lru_cache
    def p(self, gamma=1.4):
        p = self.E
        p += self.rho * self.ux * self.ux * -0.5
        p += self.rho * self.uy * self.uy * -0.5
        p *= gamma - 1
        return p

    def copy(self):
        return fluid_state(
            self.ux.copy(), self.uy.copy(), self.rho.copy(), self.E.copy()
        )

    def max_rank(self):
        max_ranks = []
        max_ranks.append(self.ux.max_rank)
        max_ranks.append(self.uy.max_rank)
        max_ranks.append(self.rho.max_rank)
        max_ranks.append(self.p.max_rank)
        max_ranks.append(self.E.max_rank)

        return max(max_ranks)

    def parameters(self):
        params = []

        params += self.ux.tensors
        params += self.uy.tensors
        params += self.rho.tensors
        params += self.E.tensors

        return params


def random_fluid_state(nofq, max_bd, requires_grad=False):
    ux = random_mps(sum(nofq), bd=max_bd, requires_grad=requires_grad)
    uy = random_mps(sum(nofq), bd=max_bd, requires_grad=requires_grad)
    rho = random_mps(sum(nofq), bd=max_bd, requires_grad=requires_grad)
    E = random_mps(sum(nofq), bd=max_bd, requires_grad=requires_grad)

    return fluid_state(ux, uy, rho, E)


def advection_term(
    state: fluid_state,
    Dx: MPO,
    Dy: MPO,
):
    advectionx = Dx.apply(state.ux * state.ux * state.rho + state.p)
    advectionx += Dy.apply(state.ux * state.uy * state.rho)

    advectiony = Dx.apply(state.uy * state.ux * state.rho)
    advectiony += Dy.apply(state.uy * state.uy * state.rho + state.p)

    return (advectionx, advectiony)


def mass_conservation(
    state: fluid_state,
    Dx: MPO,
    Dy: MPO,
):
    div = Dx.apply(state.ux * state.rho)
    div += Dy.apply(state.uy * state.rho)
    return div


def energy_conservation(
    state: fluid_state,
    Dx: MPO,
    Dy: MPO,
):
    endiv = Dx.apply(state.ux * (state.E + state.p))
    endiv += Dy.apply(state.uy * (state.E + state.p))
    return endiv


def euler_residuals(state: fluid_state, prev_state: fluid_state, Dx: MPO, Dy: MPO, dt):
    theta0 = state.rho - prev_state.rho
    theta0 *= dt
    theta0 += mass_conservation(prev_state, Dx, Dy)

    advx, advy = advection_term(prev_state, Dx, Dy)

    theta1 = state.rho * state.ux - prev_state.rho * prev_state.ux
    theta1 *= dt
    theta1 += advx

    theta2 = state.rho * state.uy - prev_state.rho * prev_state.uy
    theta2 *= dt
    theta2 += advy

    theta3 = state.E - prev_state.E
    theta3 *= dt
    theta3 += energy_conservation(prev_state, Dx, Dy)

    return theta0, theta1, theta2, theta3


def train(
    state: fluid_state,
    prev_state: fluid_state,
    loss_func: Callable,
    lr_list,
    epoch_list,
    stopping_loss,
):
    for lr, epochs in zip(lr_list, epoch_list):
        print("LR: " + str(lr))
        optimizer = torch.optim.NAdam(
            state.parameters(), lr=lr, momentum_decay=0, betas=(0.9, 0.99)
        )
        for i in range(epochs):
            optimizer.zero_grad()
            loss = loss_func(state, prev_state)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch: " + str(i))
                print("Train Loss: " + str(loss.item()))
            if loss.item() <= stopping_loss:
                break

        if loss.item() <= stopping_loss:
            print("Early Stopping: Desired Loss reached")
            break


def variational_time_stepping(
    initial_state: MPS,
    dt: float,
    T: int,
    nofq: list[int],
    max_bd: int,
    train_params: dict,
    boundary_conditions: list[str] = None,
):
    dx = parameters("bx") - parameters("ax")
    dx /= (2 ** nofq[0]) - 1
    dy = parameters("by") - parameters("ay")
    dy /= (2 ** nofq[1]) - 1
    if boundary_conditions is None:
        bc = ["periodic"] * 4
    else:
        bc = boundary_conditions

    Dx = get_diff_mpo(dim=2, order=1, axis=0, bc=bc) * (1.0 / dx)
    Dy = get_diff_mpo(dim=2, order=1, axis=1, bc=bc) * (1.0 / dy)

    def loss_func1(state, prev_state):
        t0, t1, t2, t3 = euler_residuals(state, prev_state, Dx, Dy, dt)
        loss = t0.norm**2+ t1.norm**2+ t2.norm**2 + t3.norm**2
        # loss = torch.sqrt(loss)
        loss /= 2**len(state.ux)
        return loss

    results = []

    prev_state = initial_state.copy()

    results.append(prev_state)
    print(f"starting time evolution for {T} timesteps")
    for i in tqdm(range(T)):
        state = random_fluid_state(nofq, max_bd, requires_grad=True)

        train(state, prev_state, loss_func=loss_func1, **train_params)

        prev_state = state.copy()

        results.append(prev_state)

    return results
