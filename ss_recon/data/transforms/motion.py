import torch

from ss_recon.utils import env

if env.pt_version() >= [1,6]:
    import torch.fft


class MotionModel():
    """A model that corrupts kspace inputs with motion.

    Motion is a common artifact experienced during the MR imaging forward problem.
    When a patient moves, the recorded (expected) location of the kspace sample is
    different than the actual location where the kspace sample that was acquired.

    This module is responsible for simulating different motion artifacts.

    Args:
        seed (int, optional): The fixed seed.

    Attributes:
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.

    Things to consider:
        1. What other information is relevant for inducing motion corruption?
            This could include:

            - ``traj``: The scan trajectory
            - ``etl``: The echo train length - how many readouts per shot.
            - ``num_shots``: Number of shots.
        
        2. What would a simple translational motion model look?
    
    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.
    """
    def __init__(self, seed: int = None):
        super().__init__()

        # For reproducibility. Use this generator for any random logic.
        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, mask=None, seed=None, clone=True) -> torch.Tensor:
        """Performs motion corruption on kspace image.

        TODO: The current arguments were copied from the NoiseModel.
            Feel free to change.

        Args:
            kspace (torch.Tensor): The complex tensor. Shape ``(N, Y, X, #coils, [2])``.
            mask (torch.Tensor): The undersampling mask. Shape ``(N, Y, X, #coils)``.
            seed (int, optional): Fixed seed at runtime (useful for generating testing vals).
            clone (bool, optional): If ``True``, return a cloned tensor.

        Returns:
            torch.Tensor: The motion corrupted kspace.

        Note:
            For backwards compatibility with torch<1.6, complex tensors may also have the shape
            ``(..., 2)``, where the 2 channels in the last dimension are real and
            imaginary, respectively.

            TODO: This code should account for that case as well.
        """
        raise NotImplementedError

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        """Instantiate this class from a config.

        Args:
            cfg (CfgNode): An instance of the default config.
                See ss_recon/config/defaults.py for the config structure.
            seed (int, optional): The fixed seed.
            kwargs: Keyword arguments to override cfg arguments.
        """
        raise NotImplementedError
