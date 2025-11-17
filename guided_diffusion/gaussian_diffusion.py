import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time as ti
from torch.optim import Adam
from tqdm.auto import tqdm
from torchdiffeq import odeint, odeint_adjoint
from guided_diffusion.unet import create_model

from util.img_utils import clear_color
from .posterior_mean_variance import get_mean_processor, get_var_processor
from mcmc import MCMCSampler
from precond import VPPrecond


__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing,
                   respacing_power):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing, respacing_power),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        # calculations for g_t^2
        self.g2_schedule = -self.num_timesteps * np.log(1.0 - self.betas)
        self.snr = self.sqrt_alphas_cumprod / np.sqrt(1.0 - self.alphas_cumprod)
        # self.snr[-100:] = [item / 2 for item in self.snr[-100:]]
        ############################# exact #############################
        self.alpha_schedule_std = np.sqrt(self.alphas_cumprod)
        self.alpha_T_sde = self.alpha_schedule_std[-1]            # 0.0063528
        self.inv_alpha_T_sde = 1.0 / self.alpha_T_sde
        self.total_g2_integral = -2.0 * np.log(self.alpha_T_sde)  # 10.1177
        alpha_schedule_std_rev = self.alpha_schedule_std[::-1].copy()
        self.sde_integral_term = alpha_schedule_std_rev / self.alpha_T_sde
        self.sde_denominator = self.alpha_T_sde - self.inv_alpha_T_sde  # -157.404
        ############################# EDM annealing #############################        
        self.x_start = None

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)
            
    def get_schedule_jump(self, T_sampling, travel_length, travel_repeat): # 1000    10    3

        jumps = {}
        for t in range(0, T_sampling - travel_length, travel_length):
            jumps[t] = 1

        t = T_sampling
        ts = []
        while t >= 1:
            t = t-1
            ts.append(t)
            if jumps.get(t)==1 & (t % travel_length == 0):
                jumps[t] = 0
                for _ in range(travel_repeat):
                    t = t + 1
                    ts.append(t)
        ts.append(-1)
        return ts

            

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_sample_loop_sde(self,
                          model,
                          img_idx,
                          x_start,
                          edm_sigma_max,
                          measurement,
                          operator,
                          record,
                          save_root,
                          solve_type,
                          gamma,
                          mask=None,
                          measure_config=None,
                          model_config=None,
                          results_dir=None,
                          ):
        sigma_max = edm_sigma_max
        sigma_min = 1e-1
        p=7
        time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        print('self.num_timesteps:',self.num_timesteps)
        steps = np.linspace(0, 1, self.num_timesteps+1)
        sigma_steps_list = [time_steps_fn(s) for s in steps]
        self.sigma_steps_list = torch.tensor(sigma_steps_list, dtype=torch.float32)

        l=15
        scale_max = 1.5
        scale_min = 5e-1
        scale_steps_fn = lambda r: scale_min + (scale_max - scale_min) * 1 / (1 + np.exp(-6 * (2 * r - 1)))
        scale_steps_list = [scale_steps_fn(s) for s in steps][::-1]
        self.scale_steps_list = torch.tensor(scale_steps_list, dtype=torch.float32)

        img = x_start
        device = x_start.device
        progress_images = {}
        self.x_start = x_start
        self.model_config = model_config    

        self.edm_model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(device)
        self.edm_model.eval()
        self.edm_model.requires_grad_(False)

        if solve_type == 'nonlinear_finitegamma':
            self.mcmc_config = {**measure_config['mcmc'][solve_type][gamma], **measure_config['mcmc']['param']}
            self.mcmc_sampler = MCMCSampler(**self.mcmc_config)
        elif solve_type == 'nonlinear':
            self.mcmc_config = {**measure_config['mcmc'][solve_type], **measure_config['mcmc']['param']}
            self.mcmc_sampler = MCMCSampler(**self.mcmc_config)
        
        std_history = []
        timesteps_history = []
        combined_cols = []
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            denoising_steps_taken = self.num_timesteps - 1 - idx
            current_integral = self.sde_integral_term[denoising_steps_taken]

            out = self.p_sample(model=model, x=img, t=time, y=measurement, operator=operator,
                                solve_type=solve_type, gamma = gamma, integral_value=current_integral, 
                                mask=mask, measure_config=measure_config)
            
            img = out['sample'].detach()
            show = out['show'].detach()
            x0_t_hat = out["x0_t_hat"].detach()

            if record and idx % (self.num_timesteps/10) == 0:
                combined_cols.append({
                    "idx": idx,
                    "img": clear_color(img),
                    "show": clear_color(show),
                    "x0": clear_color(x0_t_hat),
                })
                plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png"), clear_color(img))
                plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}_show.png"), clear_color(show))
                plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}_x0_t_hat.png"), clear_color(x0_t_hat))

        if record and combined_cols:
            ################################ save img ################################
            combined_cols = sorted(combined_cols, key=lambda d: d["idx"])
            combined_cols = combined_cols[:10]
            cols = len(combined_cols)
            rows = 3

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4), gridspec_kw=dict(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0))
            if cols == 1:
                axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

            for j, col in enumerate(combined_cols):
                axes[0, j].imshow(col["img"])
                # axes[0, j].set_title(f"t = {col['idx']}")
                axes[1, j].imshow(col["x0"])
                axes[2, j].imshow(col["show"])
                for r in range(rows):
                    axes[r, j].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            combined_image_path = os.path.join(results_dir, f"combined_3x10_{img_idx}.png")
            plt.savefig(combined_image_path)
            plt.close(fig)
            
        return img

        
    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    def p_sample_PFODE(self, model, x, t, num_steps=100):
        if t[0].item() == 0:
            with torch.no_grad():
                out = self.p_mean_variance(model, x, t)
                return out['pred_xstart']

        time_steps = torch.linspace(t.item(), 0, num_steps + 1, device=x.device)
        delta_t = (time_steps[0] - time_steps[1]) / self.num_timesteps

        if not hasattr(self, "betas_torch") or self.betas_torch.device != x.device:
            self.betas_torch = torch.from_numpy(self.betas).to(x.device, dtype=torch.float32)

        xt = x.clone()
        for i in range(num_steps):
            current_t_continuous = time_steps[i]
            current_t_discrete = torch.round(current_t_continuous).long()
            t_model_input = torch.full((x.shape[0],), current_t_discrete.item(), device=x.device, dtype=torch.long)

            with torch.no_grad():
                model_output = model(xt, self._scale_timesteps(t_model_input))
                if model_output.shape[1] == 2 * xt.shape[1]:
                    model_output_mean, _ = torch.split(model_output, xt.shape[1], dim=1)
                else:
                    model_output_mean = model_output
                _, pred_xstart = self.mean_processor.get_mean_and_xstart(xt, t_model_input, model_output_mean)
            
            # 在 VP-SDE 的概率流 ODE 中，导数可以简化表示为：dxt/dt = -0.5 * beta(t) * (xt + pred_xstart)
            beta_t = extract_and_expand(self.betas_torch, t_model_input, xt)
            drift = -0.5 * beta_t * (xt + pred_xstart)
            xt = xt - drift * delta_t
        x0_hat = xt
        return x0_hat

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample_torchdiffeq_edm(self, edm_model, x, t, sigma_max, num_steps=5, solver='euler', requires_grad=False):
        sigma_min = 1e-2
        p = 7
        time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        steps = np.linspace(0, 1, num_steps+1)
        time_steps_list = [time_steps_fn(s) for s in steps]
        time_steps = torch.tensor(time_steps_list, device=x.device, dtype=torch.float32)
        original_shape = x.shape

        def _ode_derivative(sigma_ode, x_ode):
            x_ode = x_ode.view(original_shape)
            with torch.no_grad():
                model_output_mean = edm_model.forward(x_ode, torch.as_tensor(sigma_ode).to(x_ode.device))
                
            drift = -(model_output_mean - x_ode) / sigma_ode

            return drift.flatten(1)
        
        ode_solver = odeint_adjoint if requires_grad else odeint
        with torch.set_grad_enabled(requires_grad):
            solution_traj = ode_solver(
                _ode_derivative, x.flatten(1), time_steps, method=solver, rtol=1e-3, atol=1e-3
            )
        x0_hat = solution_traj[-1].view(original_shape)
        return x0_hat

    # (x_t / sqrt(ᾱ_t) - x̂₀) / sqrt(1/ᾱ_t - 1)
    def _predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


def space_timesteps(num_timesteps, section_counts, respacing_power):
    u = np.linspace(0.0, 1.0, int(section_counts))
    t_cont = ((1.0 - u) ** respacing_power) * (num_timesteps - 1)
    t_idx = np.rint(t_cont).astype(np.int64)
    t_idx = np.clip(t_idx, 0, num_timesteps - 1)
    t_idx = np.unique(t_idx)
    t_idx.sort()
    t_idx = t_idx[::-1].copy()

    while len(t_idx) < int(section_counts):
        inserted = []
        for a, b in zip(t_idx[:-1], t_idx[1:]):
            if len(t_idx) + len(inserted) >= int(section_counts):
                break
            mid = (a + b) // 2
            if mid != a and mid != b:
                inserted.append(mid)
        if not inserted:
            if t_idx.size and t_idx[0] < (T - 1):
                inserted.append(min(T - 1, t_idx[0] + 1))
            if t_idx.size and t_idx[-1] > 0 and len(t_idx) + len(inserted) < int(section_counts):
                inserted.append(max(0, t_idx[-1] - 1))
            if not inserted:
                break
        t_idx = np.unique(np.concatenate([t_idx, np.array(inserted, dtype=np.int64)]))
        t_idx.sort()
        t_idx = t_idx[::-1].copy()

    return set(t_idx.tolist())




class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm_sde')
class SDESampler(SpacedDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(use_timesteps, **kwargs)
        alphas_cumprod_tensor = torch.from_numpy(self.alphas_cumprod).float()
        sigma2 = (1.0 - alphas_cumprod_tensor) / (alphas_cumprod_tensor + 1e-9)
        sigma_steps_torch = torch.sqrt(torch.where(sigma2 < 0, torch.zeros_like(sigma2), sigma2))
        self.sigma_steps = sigma_steps_torch.numpy()[::-1].copy()
        self.diffusion_scheduler_config = {
            'name': 'edm',
            'num_steps': 5,
            'sigma_min': 1e-2,
            'timestep': 'poly-7'
        }

    
    def Solve_with_MCMC(self, model, xt, t, pred_xstart, operator, measurement):
        time_step = t.item()
        ratio = ((self.num_timesteps - 1 - time_step) / (self.num_timesteps - 1)) if self.num_timesteps > 1 else 1.0
        x0_t_hat = self.mcmc_sampler.sample(xt=xt, model=model, x0hat=pred_xstart, operator=operator, 
                                       measurement=measurement, sigma=self.sigma_steps_list[self.num_timesteps - 1 - time_step], ratio=ratio)

        return x0_t_hat


    def Solve_with_MCMC_finitegamma(self, model, xt, t, pred_xstart, operator, measurement, gamma):
        time_step = t.item()
        ratio = ((self.num_timesteps - 1 - time_step) / (self.num_timesteps - 1)) if self.num_timesteps > 1 else 1.0
        c_scalar = gamma * (1.0 - np.exp(-self.total_g2_integral))
        k_scalar = np.exp(-0.5 * self.total_g2_integral)

        x_solution = self.mcmc_sampler.sample_finitegamma(xt=xt, model=model, pred_xstart=pred_xstart, operator=operator, measurement=measurement, sigma=self.sigma_steps_list[self.num_timesteps - 1 - time_step], ratio=ratio, c_scalar=c_scalar, k_scalar=k_scalar, gamma=gamma, x_start=self.x_start)
        
        return x_solution


    def p_sample(self, model, x, t, y, operator, solve_type, gamma, integral_value, mask=None, measure_config=None):
        out = self.p_mean_variance(model, x, t)
        pred_xstart = out['pred_xstart']

        g2_t = torch.full((x.shape[0],), self.g2_schedule[t], device=x.device)
        g2_t = g2_t.view(-1, 1, 1, 1)
        # exp(1/2 * integral from 0 to t_sde)
        inv_alpha_t_sde = integral_value
        dt = (1/ self.num_timesteps)

        if solve_type == 'linear':
            at = np.sqrt(self.alphas_cumprod[t])
            at_next = np.sqrt(self.alphas_cumprod_prev[t])
            sigma = np.sqrt(1.0 - self.alphas_cumprod_prev[t])
            sigma_y = 0.05*2
            lambda_t = 1 if sigma >= at_next * sigma_y else (sigma) / (at_next * sigma_y)
            if measure_config['operator']['name'] in ['nonlinear_blur_soc', 'high_dynamic_range_soc']:
                pred_xstart = pred_xstart - lambda_t * operator.transpose(operator.forward(pred_xstart) - y)
                
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                x0_t_hat = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=4, solver='euler')
            else:
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                pred_xstart = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=4, solver='euler')
                x0_t_hat = pred_xstart - lambda_t * operator.transpose(operator.forward(pred_xstart) - y)

            A_t = (1.0 / inv_alpha_t_sde) * (self.inv_alpha_T_sde ** 2 - inv_alpha_t_sde ** 2) / (self.inv_alpha_T_sde ** 2 - 1)
            B_t = (inv_alpha_t_sde - 1.0 / inv_alpha_t_sde) / (self.inv_alpha_T_sde - self.alpha_T_sde)

            sigma_t = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x0_t_hat)
            noise = torch.randn_like(x0_t_hat)
            sample = A_t * self.x_start + B_t * x0_t_hat + (torch.exp(0.5 * out['log_variance']) - A_t) * noise

        elif solve_type == 'nonlinear':
            if t[0].item() != 0:
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                pred_xstart = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=4, solver='euler')
            x0_t_hat = self.Solve_with_MCMC(model, x, t, pred_xstart, operator, y)

            A_t = (1.0 / inv_alpha_t_sde) * (self.inv_alpha_T_sde ** 2 - inv_alpha_t_sde ** 2) / (self.inv_alpha_T_sde ** 2 - 1)
            B_t = (inv_alpha_t_sde - 1.0 / inv_alpha_t_sde) / (self.inv_alpha_T_sde - self.alpha_T_sde)

            sigma_t = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x0_t_hat)
            noise = torch.randn_like(x0_t_hat)
            sample = A_t * self.x_start + B_t * x0_t_hat + (sigma_t - A_t)* noise

        elif solve_type == 'nonlinear_finitegamma':
            if t[0].item() != 0:
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                pred_xstart = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=4, solver='euler')
            x0_t_hat = self.Solve_with_MCMC_finitegamma(model, x, t, pred_xstart, operator, y, gamma=gamma)

            sigma_t = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x0_t_hat)
            noise = torch.randn_like(x0_t_hat)
            if measure_config['operator']['name'] in ['high_dynamic_range_soc']:
                sample = (1.0 / inv_alpha_t_sde) * self.x_start + self.alpha_T_sde * (inv_alpha_t_sde - 1.0 / inv_alpha_t_sde) * x0_t_hat + (torch.exp(0.5 * out['log_variance']) - (1.0 / inv_alpha_t_sde)) * noise
            else:
                sample = (1.0 / inv_alpha_t_sde) * self.x_start + self.alpha_T_sde * (inv_alpha_t_sde - 1.0 / inv_alpha_t_sde) * x0_t_hat
        

        return {"sample": sample, "pred_xstart": pred_xstart, "show": pred_xstart, "x0_t_hat": x0_t_hat}


# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    if not isinstance(array, torch.Tensor):
        tensor_array = torch.from_numpy(array)
    else:
        tensor_array = array
    array = tensor_array.to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
