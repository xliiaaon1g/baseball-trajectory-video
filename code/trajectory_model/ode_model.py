"""Small acceleration network and differentiable RK4 in SI units.

Inference accepts ONLY an initial 6D state, pitch type and elapsed timestamps.
"""
import math

import torch
from torch import nn


class BallODE(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.register_buffer('state_mean', torch.tensor(normalization['state_mean'], dtype=torch.float64))
        self.register_buffer('state_std', torch.tensor(normalization['state_std'], dtype=torch.float64))
        self.register_buffer('gravity', torch.tensor([0, 0, -9.81], dtype=torch.float64))
        self.register_buffer('acceleration_scale', torch.tensor(10., dtype=torch.float64))
        self.network = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3)).double()
        # Start with a small non-gravity output; every force correction is learned.
        nn.init.normal_(self.network[-1].weight, mean=0, std=.01)
        nn.init.zeros_(self.network[-1].bias)

    def non_gravity_acceleration(self, state, pitch_onehot):
        normalized = (state-self.state_mean)/self.state_std
        return self.network(torch.cat([normalized, pitch_onehot], dim=-1))*self.acceleration_scale

    def forward(self, state, pitch_onehot):
        return torch.cat([state[..., 3:], self.gravity+self.non_gravity_acceleration(state, pitch_onehot)], dim=-1)


def rk4_step(model, state, pitch_onehot, dt):
    h = dt[:, None]
    k1 = model(state, pitch_onehot)
    k2 = model(state+h*k1/2, pitch_onehot)
    k3 = model(state+h*k2/2, pitch_onehot)
    k4 = model(state+h*k3, pitch_onehot)
    return state+h*(k1+2*k2+2*k3+k4)/6


def integrate(model, initial_state, pitch_onehot, elapsed_times, max_step=1/240):
    """Batched exact timestamp sampling, shortened final steps, padded times allowed."""
    assert elapsed_times.ndim == 2 and initial_state.shape[-1] == 6
    assert torch.all(elapsed_times[:, 0] == 0)
    state = initial_state
    states = [state]
    for index in range(1, elapsed_times.shape[1]):
        dt = elapsed_times[:, index]-elapsed_times[:, index-1]
        assert torch.all(dt >= 0)
        steps = max(1, math.ceil(float(dt.max())/max_step-1e-12))
        for _ in range(steps):
            state = rk4_step(model, state, pitch_onehot, dt/steps)
        states.append(state)
    return torch.stack(states, dim=1)
