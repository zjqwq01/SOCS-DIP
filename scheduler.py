from torch.autograd import grad
from abc import ABC, abstractmethod
from torchdiffeq import odeint, odeint_adjoint
from precond import VPPrecond
from guided_diffusion.unet import create_model
import numpy as np
import tqdm
import torch
import warnings

__DIFFUSION_SCHEDULER__ = {}


def register_diffusion_scheduler(name: str):
    def wrapper(cls):
        if __DIFFUSION_SCHEDULER__.get(name, None):
            if __DIFFUSION_SCHEDULER__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DIFFUSION_SCHEDULER__[name] = cls
        cls.name = name
        return cls

    return wrapper

def get_diffusion_scheduler(name: str, **kwargs):
    if __DIFFUSION_SCHEDULER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DIFFUSION_SCHEDULER__[name](**kwargs)


class Scheduler(ABC):
    """
    Abstract base class for diffusion scheduler.

    Schedulers manage time steps, noise scales (sigma), scaling factors, and coefficients 
    used in diffusion stochastic/ordinary differential equations (SDEs/ODEs).
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps + 1 # include the initial step

    def discretize(self, time_steps):
        sigma_steps = self.get_sigma(time_steps[:-1])
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        self.sigma_steps = sigma_steps

    def tensorize(self, data):
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, list):
            return torch.tensor(data).float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        raise ValueError(f"Data type {type(data)} is not supported.") 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Noise Scheduling & Scaling Function 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_scaling(self, t):
        pass
    
    def get_sigma(self, t):
        pass
    
    def get_scaling_derivative(self, t):
        pass

    def get_sigma_derivative(self, t):
        pass

    def get_sigma_inv(self, sigma):
        pass
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Time & Sigma Range Function
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_t_min(self):
        pass

    def get_t_max(self):
        pass

    def get_discrete_time_steps(self, num_steps):
        pass

    def get_sigma_max(self):
        return self.get_sigma(self.get_t_max())

    def get_sigma_min(self):
        return self.get_sigma(self.get_t_min())
    
    def get_prior_sigma(self):
        # simga(t_max) * scaling(t_max)
        return self.get_sigma_max() * self.get_scaling(self.get_t_max())

    def summary(self):
        print('+' * 50)
        print('Diffusion Scheduler Summary')
        print('+' * 50)
        print(f"Scheduler       : {self.name}")
        print(f"Time Range      : [{self.get_t_min().item()}, {self.get_t_max().item()}]")
        print(f"Sigma Range     : [{self.get_sigma_min().item()}, {self.get_sigma_max().item()}]")
        print(f"Scaling Range   : [{self.get_scaling(self.get_t_min()).item()}, {self.get_scaling(self.get_t_max()).item()}]")
        print(f"Prior Sigma     : {self.get_prior_sigma().item()}")
        print('+' * 50)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For Iterating Over the Discretized Scheduler
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __iter__(self):
        self.pbar = tqdm.trange(self.num_steps) if self.verbose else range(self.num_steps)
        self.pbar_iter = iter(self.pbar)
        return self

    def __next__(self):
        try:
            step = next(self.pbar_iter)
            time, scaling, sigma, scaling_factor, factor = self.time_steps[step], self.scaling_steps[step], \
                self.sigma_steps[step], self.scaling_factor_steps[step], self.factor_steps[step]
            return self.pbar, time, scaling, sigma, factor, scaling_factor
        except StopIteration:
            raise StopIteration


@register_diffusion_scheduler('edm')
class EDMScheduler(Scheduler):
    """
        EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Scheduler.
    """

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2, timestep='poly-7'):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        p = int(timestep.split('-')[1])
        self.time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = t
        return self.tensorize(t)

    def get_scaling(self, t):
        # s(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        return torch.zeros_like(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return self.tensorize(sigma)

    def get_t_min(self):
        return self.tensorize(self.sigma_min)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max)

    def get_discrete_time_steps(self, num_steps):
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([self.time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps)


class DiffusionPFODE:
    def __init__(self, model, scheduler, model_config, solver='euler'):
        self.scheduler = scheduler
        self.solver = solver
        self.device = next(model.parameters()).device
        self.edm_model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(self.device)
        self.edm_model.eval()
        self.edm_model.requires_grad_(False)

    def sample(self, xT, num_steps=None, return_traj=False, requires_grad=False):
        if num_steps is None:
            num_steps = self.scheduler.num_steps
        shape = xT.shape

        def _derivative_wrapper(sigma_ode, x_ode):
            x_ode = x_ode.view(shape)
            d = self.edm_model.forward(x_ode, torch.as_tensor(sigma_ode).to(x_ode.device))
            score = (d - x_ode) / sigma_ode ** 2
            drift = - sigma_ode * score
            # drift = - sigma_ode * self.model.score(x_ode, sigma=sigma_ode)
            return drift.flatten(1)

        time_steps = self.scheduler.get_discrete_time_steps(num_steps).to(xT.device)
        print('time_steps:',time_steps)
        if requires_grad:
            xT.requires_grad_(True)
            x_ode_traj = odeint_adjoint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver, adjoint_params=(xT))
        else:
            x_ode_traj = odeint(_derivative_wrapper, xT.flatten(1), time_steps, rtol=1e-3, atol=1e-3, method=self.solver)
        
        x_ode_traj = x_ode_traj.view(num_steps, *shape)
        
        if return_traj:
            return x_ode_traj
        else:
            return x_ode_traj[-1]