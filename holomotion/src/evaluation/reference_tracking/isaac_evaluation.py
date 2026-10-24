"""The unchanged full-clip scoring loop with independent Isaac physics."""

from holomotion.src.evaluation.reference_tracking.evaluation import (
    FullClipEvaluator,
)


class IsaacFullClipEvaluator(FullClipEvaluator):
    """Defer physics until final evaluation, after training has finished."""

    def __init__(
        self,
        model,
        motion,
        mdp,
        *,
        device,
        history_length=10,
        future_frames=10,
        capture_graph=True,
        robot_urdf=None,
        cache_dir=None,
    ):
        self.evaluation_mode = "native"
        self.motion, self.mdp = motion, mdp
        self.history_length, self.future_frames = history_length, future_frames
        self.model, self.device = model, device
        self.robot_urdf, self.cache_dir = robot_urdf, cache_dir
        self._env = None

    @property
    def env(self):
        if self._env is None:
            from holomotion.src.env.reference_tracking.isaac_backend import (
                IsaacTrackingBackend,
            )

            self._env = IsaacTrackingBackend(
                self.model,
                self.motion,
                self.mdp,
                len(self.motion.names),
                device=self.device,
                evaluation=True,
                capture_graph=False,
                robot_urdf=self.robot_urdf,
                cache_dir=self.cache_dir,
            )
        return self._env
