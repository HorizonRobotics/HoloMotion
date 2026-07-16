import torch

from holomotion.src.algo.ppo_tf import PPOTF


def _sos():
    return PPOTF._design_butterworth_highpass_sos(
        sample_hz=50.0,
        cutoff_hz=3.0,
        order=4,
    )


def _loss(action_mean: torch.Tensor, dones: torch.Tensor | None = None):
    batch_size, sequence_length, _ = action_mean.shape
    if dones is None:
        dones = torch.zeros(
            batch_size,
            sequence_length,
            1,
            device=action_mean.device,
            dtype=torch.bool,
        )
    valid_tok = torch.ones(
        batch_size,
        sequence_length,
        device=action_mean.device,
    )
    return PPOTF._action_mean_butterworth_loss(
        action_mean=action_mean,
        dones=dones,
        valid_tok=valid_tok,
        sos=_sos(),
    )


def test_action_mean_butterworth_aux_is_differentiable():
    time = torch.arange(64, dtype=torch.float32) / 50.0
    action_mean = torch.sin(2.0 * torch.pi * 8.0 * time)[None, :, None]
    action_mean.requires_grad_()

    loss = _loss(action_mean)
    loss.backward()

    assert loss.item() > 0.0
    assert action_mean.grad is not None
    assert torch.count_nonzero(action_mean.grad).item() > 0


def test_action_mean_butterworth_aux_rejects_low_frequency_energy():
    time = torch.arange(128, dtype=torch.float32) / 50.0
    low = torch.sin(2.0 * torch.pi * 1.0 * time)[None, :, None]
    high = torch.sin(2.0 * torch.pi * 8.0 * time)[None, :, None]

    low_loss = _loss(low)
    high_loss = _loss(high)

    assert high_loss > 100.0 * low_loss


def test_action_mean_butterworth_aux_resets_at_episode_boundary():
    action_mean = torch.cat(
        (
            -torch.ones(1, 16, 1),
            torch.ones(1, 16, 1),
        ),
        dim=1,
    )
    dones = torch.zeros(1, 32, 1, dtype=torch.bool)
    dones[:, 15] = True

    reset_loss = _loss(action_mean, dones)
    continuous_loss = _loss(action_mean)

    assert reset_loss < 1.0e-10
    assert continuous_loss > 1.0e-3
