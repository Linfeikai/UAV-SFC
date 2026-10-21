"""Learned complete-deployment scoring with an explicit shared braking controller."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algos.placement_candidates import CandidateGeometry, placement_candidates, candidate_features


class DeploymentScorer(nn.Module):
    def __init__(self, dimensions, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class ScoredDeploymentPolicy:
    """A learned scorer chooses a whole proposal from observation-derived features.

    Movement is deliberately an analytical velocity-braking controller. This
    family learns deployment selection; it does not claim learned mobility.
    No environment or simulator is supplied to prediction.
    """
    def __init__(self, model, mean, scale, geometry=CandidateGeometry(), margin=0., device='cpu'):
        self.model = model.to(device).eval()
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.geometry, self.margin, self.device = geometry, float(margin), device

    def predict(self, obs, deterministic=True):
        proposals = placement_candidates(obs, self.geometry)
        features = candidate_features(obs, proposals, self.geometry)
        with torch.no_grad():
            x = (torch.as_tensor(features, device=self.device) - self.mean) / self.scale
            values = self.model(x).cpu().numpy() * 3.0
        index = int(values.argmax())
        if values[index] - values[0] < self.margin:
            index = 0
        return proposals[index], None

    def save(self, path, **metadata):
        torch.save(dict(state_dict=self.model.cpu().state_dict(), dimensions=self.mean.numel(),
                        hidden=self.model.layers[0].out_features, mean=self.mean.cpu(),
                        scale=self.scale.cpu(), geometry=asdict(self.geometry), margin=self.margin,
                        metadata=metadata), Path(path))
        self.model.to(self.device)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device, weights_only=True)
        model = DeploymentScorer(data['dimensions'], data['hidden'])
        model.load_state_dict(data['state_dict'])
        return cls(model, data['mean'], data['scale'], CandidateGeometry(**data['geometry']),
                   data['margin'], device)
