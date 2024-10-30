# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Math Helper Functions """

from dataclasses import dataclass

import torch
from torchtyping import TensorType


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: TensorType[..., "dim"]
    cov: TensorType[..., "dim", "dim"]


def compute_3d_gaussian(
    directions: TensorType[..., 3],
    means: TensorType[..., 3],
    dir_variance: TensorType[..., 1],
    radius_variance: TensorType[..., 1],
) -> Gaussians:
    """Compute guassian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


def cylinder_to_gaussian(
    origins: TensorType[..., 3],
    directions: TensorType[..., 3],
    starts: TensorType[..., 1],
    ends: TensorType[..., 1],
    radius: TensorType[..., 1],
) -> Gaussians:
    """Approximates cylinders with a Gaussian distributions.

    Args:
        origins: Origins of cylinders.
        directions: Direction (axis) of cylinders.
        starts: Start of cylinders.
        ends: End of cylinders.
        radius: Radii of cylinders.

    Returns:
        Gaussians: Approximation of cylinders
    """
    means = origins + directions * ((starts + ends) / 2.0)
    dir_variance = (ends - starts) ** 2 / 12
    radius_variance = radius**2 / 4.0
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def conical_frustum_to_gaussian(
    origins: TensorType[..., 3],
    directions: TensorType[..., 3],
    starts: TensorType[..., 1],
    ends: TensorType[..., 1],
    radius: TensorType[..., 1],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    radius_variance = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)
