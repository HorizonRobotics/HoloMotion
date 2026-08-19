import unittest
from unittest.mock import patch

from holoretarget._engine_impl import (
    HoloNewtonG1Retargeter,
    HoloRetargetRunner,
)
from holoretarget.config import DEFAULT_CONFIG
from holoretarget.online import HoloRetargeter
from holoretarget.schema import DOF_POS_DIM


class _FakeIkSolver:
    def __init__(self):
        self.calls = []

    def step(self, joint_q_in, joint_q_out, *, iterations, step_size):
        self.calls.append(
            {
                "joint_q_in": joint_q_in,
                "joint_q_out": joint_q_out,
                "iterations": iterations,
                "step_size": step_size,
            }
        )


class HoloRetargetNewtonTests(unittest.TestCase):
    def test_production_step_size_is_damped(self):
        self.assertEqual(DEFAULT_CONFIG.newton_step_size, 0.75)

    def test_single_step_forwards_configured_step_size(self):
        retargeter = HoloNewtonG1Retargeter.__new__(HoloNewtonG1Retargeter)
        retargeter.ik_solver = _FakeIkSolver()
        retargeter.joint_q = object()
        retargeter.ik_iterations = 2
        retargeter.step_size = 0.75

        retargeter._single_step()

        self.assertEqual(len(retargeter.ik_solver.calls), 1)
        call = retargeter.ik_solver.calls[0]
        self.assertIs(call["joint_q_in"], retargeter.joint_q)
        self.assertIs(call["joint_q_out"], retargeter.joint_q)
        self.assertEqual(call["iterations"], 2)
        self.assertEqual(call["step_size"], 0.75)

    def test_online_config_reaches_runner(self):
        retargeter = HoloRetargeter.__new__(HoloRetargeter)
        retargeter.config = DEFAULT_CONFIG

        with patch("holoretarget.online.HoloRetargetRunner") as runner_class:
            retargeter._create_runner(src_human="smplx")

        self.assertEqual(
            runner_class.call_args.kwargs["newton_step_size"],
            0.75,
        )

    def test_runner_forwards_step_size_to_newton_solver(self):
        with patch(
            "holoretarget._engine_impl.HoloNewtonG1Retargeter"
        ) as solver_class:
            solver_class.return_value.num_dof = DOF_POS_DIM
            HoloRetargetRunner(newton_step_size=0.75)

        self.assertEqual(solver_class.call_args.kwargs["step_size"], 0.75)


if __name__ == "__main__":
    unittest.main()
