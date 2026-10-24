"""Independent boundary checks for the simulator PPO comparison."""

import copy

import torch

from holomotion.src.algo.reference_ppo.learner import (
    SimulatorPPOLearner,
    clipped_losses,
    generalized_advantage,
)


def test_timeout_bootstraps_physical_next_state_but_does_not_cross_reset():
    rewards = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
    terminated = torch.tensor([[False, True], [False, False]])
    ended = torch.tensor([[True, True], [False, False]])
    advantage, targets = generalized_advantage(
        rewards, values, next_values, terminated, ended, gamma=0.5, lam=1.0
    )
    torch.testing.assert_close(
        advantage, torch.tensor([[2.0, 1.0], [100.0, 100.0]])
    )
    torch.testing.assert_close(targets, advantage)
    continued, _ = generalized_advantage(
        rewards,
        values,
        next_values,
        terminated,
        torch.zeros_like(ended),
        gamma=0.5,
        lam=1.0,
    )
    assert continued[0, 0] == 52.0


def test_clipping_limits_both_signs_and_uses_clipped_value_maximum():
    log_ratio = torch.tensor([2.0, 0.5]).log()
    loss, policy, value = clipped_losses(
        log_ratio,
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(2),
    )
    torch.testing.assert_close(policy, torch.tensor(-0.2))
    torch.testing.assert_close(value, torch.tensor(0.64))
    torch.testing.assert_close(loss, torch.tensor(0.44))


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(5, 29)

    def forward(self, history, goal):
        return self.net(torch.cat((history.flatten(1), goal), -1))


def test_ppo_microbatches_preserve_logical_minibatch_update():
    torch.manual_seed(82)
    learner = SimulatorPPOLearner.__new__(SimulatorPPOLearner)
    torch.nn.Module.__init__(learner)
    learner.pi, learner.critic = (
        TinyPolicy(),
        torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0)),
    )
    learner.log_std = torch.nn.Parameter(torch.zeros(29))
    learner.epochs, learner.minibatches, learner.microbatch = 1, 1, 4
    learner.learning_rate = 0.001
    learner.optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)
    other = copy.deepcopy(learner)
    other.microbatch = 1
    history, goal, features = (
        torch.randn(4, 1, 2),
        torch.randn(4, 3),
        torch.randn(4, 3),
    )
    with torch.no_grad():
        dist = learner.distribution(history, goal)
        action = dist.sample()
        rollout = dict(
            history=history[None],
            goal=goal[None],
            critic_input=features[None],
            sampled_action=action[None],
            log_probability=dist.log_prob(action).sum(-1)[None],
            mean=dist.loc[None],
            std=dist.scale[None],
            value=learner.critic(features)[None],
            reward=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            next_value=torch.zeros(1, 4),
            terminated=torch.ones(1, 4, dtype=torch.bool),
            ended=torch.ones(1, 4, dtype=torch.bool),
        )
    before = learner.pi.net.weight.detach().clone()
    torch.manual_seed(2)
    result = learner.update(rollout)
    torch.manual_seed(2)
    other.update(rollout)
    assert result["optimizer_steps"] == 1
    assert not torch.equal(before, learner.pi.net.weight)
    for one, two in zip(learner.parameters(), other.parameters()):
        torch.testing.assert_close(one, two, atol=2e-7, rtol=2e-6)


def test_actor_and_critic_initialization_match_the_model_based_control():
    from types import SimpleNamespace

    from holomotion.src.algo.supermimic.learner import ReferenceLearner
    from holomotion.src.algo.supermimic.networks import WorldModel

    mdp = SimpleNamespace(obs_dim=93)
    args = SimpleNamespace(
        device="cpu",
        goal_frames=10,
        hist=10,
        horizon=32,
        lr=0.0002,
        compile=False,
    )
    torch.manual_seed(0)
    reference = ReferenceLearner(args, None, mdp, None)
    torch.manual_seed(0)
    unused = WorldModel()
    del unused
    control = SimulatorPPOLearner(mdp, None, device="cpu")
    for name in ("pi", "critic"):
        expected = getattr(reference, name).state_dict()
        actual = getattr(control, name).state_dict()
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[key], actual[key]) for key in expected)
