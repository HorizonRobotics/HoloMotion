from __future__ import annotations
import copy

import torch

from holomotion.src.modules.network_modules import MLP


def test_mlp_checkpointing_preserves_forward_and_gradients() -> None:
    torch.manual_seed(7)
    config = {
        "hidden_norm": "rmsnorm",
        "layer_config": {
            "hidden_dims": [32, 32, 32],
            "activation": "SiLU",
        },
    }
    reference = MLP(17, 5, config).train()
    checkpointed = MLP(
        17,
        5,
        {**config, "use_checkpointing": True},
    ).train()
    checkpointed.load_state_dict(copy.deepcopy(reference.state_dict()))

    reference_input = torch.randn(23, 17, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    checkpointed_output = checkpointed(checkpointed_input)
    torch.testing.assert_close(
        checkpointed_output,
        reference_output,
        rtol=0,
        atol=0,
    )

    reference_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    torch.testing.assert_close(
        checkpointed_input.grad,
        reference_input.grad,
        rtol=0,
        atol=0,
    )
    for reference_param, checkpointed_param in zip(
        reference.parameters(),
        checkpointed.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            checkpointed_param.grad,
            reference_param.grad,
            rtol=0,
            atol=0,
        )
