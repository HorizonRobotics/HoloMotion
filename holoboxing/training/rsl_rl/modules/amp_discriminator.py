# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) Discriminator Module."""

import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Discriminator network for AMP (Adversarial Motion Priors).

    The discriminator learns to distinguish between agent motion and reference motion.
    It takes AMP observations as input and outputs a scalar logit indicating
    whether the motion is from the reference dataset (positive) or agent (negative).

    Args:
        input_dim: Dimension of AMP observations
        hidden_dims: List of hidden layer dimensions (default: [1024, 512])
        activation: Activation function (default: ELU)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        activation: str = "elu",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        # Build discriminator network
        layers = []
        prev_dim = input_dim

        # Activation function
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Output layer (single logit)
        layers.append(nn.Linear(prev_dim, 1))

        self.trunk = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights following SMPLOlympics convention."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize hidden layers with default initialization
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize output layer with uniform distribution (SMPLOlympics style)
        # This gives initial logits close to 0, meaning uncertain predictions
        nn.init.uniform_(self.trunk[-1].weight, -1.0, 1.0)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of discriminator.

        Args:
            amp_obs: AMP observations [batch_size, obs_dim]

        Returns:
            logits: Discriminator logits [batch_size, 1]
                   Higher values indicate more "reference-like" motion
        """
        return self.trunk(amp_obs)

    def compute_grad_penalty(
        self, demo_obs: torch.Tensor, lambda_: float = 5.0
    ) -> torch.Tensor:
        """Compute gradient penalty following SMPLOlympics style.

        Computes the L2 norm of gradients w.r.t. demo observations,
        which encourages the discriminator to have bounded gradients.

        Args:
            demo_obs: Demo (reference) observations [batch_size, obs_dim]
            lambda_: Gradient penalty coefficient

        Returns:
            grad_penalty: Gradient penalty loss
        """
        demo_obs.requires_grad_(True)

        # Discriminator output on demo data
        disc_demo_logits = self.forward(demo_obs)

        # Compute gradients w.r.t. demo observations
        gradients = torch.autograd.grad(
            outputs=disc_demo_logits,
            inputs=demo_obs,
            grad_outputs=torch.ones_like(disc_demo_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute squared L2 norm of gradients
        # Shape: [batch_size, obs_dim] -> [batch_size]
        grad_squared = torch.sum(torch.square(gradients), dim=-1)

        # Mean over batch
        grad_penalty = lambda_ * torch.mean(grad_squared)

        return grad_penalty
