import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
from torchdiffeq import odeint, odeint_adjoint
from guided_diffusion.unet import create_model

from util.img_utils import clear_color
from .posterior_mean_variance import get_mean_processor, get_var_processor
from scipy.sparse.linalg import cg, LinearOperator, gmres, bicgstab
from torch.autograd.functional import jvp
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
                   timestep_respacing=""):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
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

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = x_start.device

        std_history = []
        timesteps_history = []

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            
            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'])
            img = img.detach_()
            std_history.append(img.std().item())
            timesteps_history.append(idx)
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 100 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))
        
        fig_std, ax_std = plt.subplots(figsize=(10, 6))
        ax_std.plot(timesteps_history[::-1], std_history[::-1])
        ax_std.set_xlabel("Timestep (t)")
        ax_std.set_ylabel("Standard Deviation of img tensor")
        ax_std.set_title("Image Tensor Standard Deviation During Denoising")
        ax_std.grid(True)
        std_plot_path = os.path.join(save_root, f"progress/std_vs_timestep.png")
        plt.savefig(std_plot_path)
        plt.close(fig_std)

        # fig_std, ax_std = plt.subplots(figsize=(10, 6))
        # ax_std.plot(timesteps_history[::-1], self.snr[::-1])
        # ax_std.set_xlabel("Timestep (t)")
        # ax_std.set_ylabel("SNR")
        # ax_std.set_title("Image SNR During Denoising")
        # ax_std.grid(True)
        # std_plot_path = os.path.join(save_root, f"progress/SNR_vs_timestep.png")
        # plt.savefig(std_plot_path)
        # plt.close(fig_std)
        
        return img


    def p_sample_loop_sde(self,
                          model,
                          x_start,
                          img_idx,
                          edm_sigma_max,
                          measurement,
                          operator,
                          record,
                          save_root,
                          solve_type,
                          gamma,
                          time_travel,
                          mask=None,
                          measure_config=None,
                          model_config=None,
                          results_dir=None,
                          ):
        sigma_max = edm_sigma_max
        sigma_min = 1e-1
        p=7
        time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        steps = np.linspace(0, 1, self.num_timesteps+1)
        sigma_steps_list = [time_steps_fn(s) for s in steps]
        self.sigma_steps_list = torch.tensor(sigma_steps_list, dtype=torch.float32)

        l=15
        scale_max = 1.5
        scale_min = 5e-1
        scale_steps_fn = lambda r: scale_min + (scale_max - scale_min) * 1 / (1 + np.exp(-6 * (2 * r - 1)))
        # scale_steps_fn = lambda r: (scale_max ** (1 / l) + r * (scale_min ** (1 / l) - scale_max ** (1 / l))) ** l
        scale_steps_list = [scale_steps_fn(s) for s in steps][::-1]
        self.scale_steps_list = torch.tensor(scale_steps_list, dtype=torch.float32)
        # print('self.scale_steps_list:',self.scale_steps_list)

        img = x_start
        device = x_start.device
        progress_images = {}
        self.x_start = x_start
        self.model_config = model_config

        # self.p_mean = self.p_mean_variance(model, x_start, torch.tensor([self.num_timesteps - 1] * img.shape[0], device=device))
        

        ############################# can be delete #############################
        self.edm_model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(device)
        self.edm_model.eval()
        self.edm_model.requires_grad_(False)
        #########################################################################

        if solve_type == 'nonlinear_finitegamma' and measure_config['operator']['name'] != 'phase_retrieval_soc':
            self.mcmc_config = {**measure_config['mcmc'][solve_type][gamma], **measure_config['mcmc']['param']}
            
            self.mcmc_sampler = MCMCSampler(**self.mcmc_config)
        elif solve_type == 'nonlinear':
            self.mcmc_config = {**measure_config['mcmc'][solve_type], **measure_config['mcmc']['param']}
            self.mcmc_sampler = MCMCSampler(**self.mcmc_config)
        

        
        if time_travel == True:
            print("Time Traveling!")
            times = self.get_schedule_jump(T_sampling=self.num_timesteps, travel_length=30, travel_repeat=1)
            time_pairs = list(zip(times[:-1], times[1:]))
            pbar = tqdm(time_pairs)
            for i, j in pbar:
                if j < i:
                    time = torch.tensor([i] * img.shape[0], device=device)
                    denoising_steps_taken = self.num_timesteps - 1 - i
                    current_integral = self.sde_integral_term[denoising_steps_taken]
                    
                    out = self.p_sample(model=model, x=img, t=time, y=measurement, operator=operator, 
                                        solve_type=solve_type, gamma=gamma, integral_value=current_integral, 
                                        mask=mask, measure_config=measure_config)
                    
                    img = out['sample'].detach()
                    pred_x0 = out['pred_xstart'].detach()
                else:
                    time_j = torch.tensor([j] * img.shape[0], device=device)
                    img = self.q_sample(x_start=pred_x0, t=time_j)

                if record and i % 100 == 0:
                    plt.imsave(os.path.join(save_root, f"progress/x_{str(i).zfill(4)}.png"), clear_color(img))
                    progress_images[i] = clear_color(img)
        else:
            print("Not Time Traveling!")

            std_history = []
            timesteps_history = []

            pbar = tqdm(list(range(self.num_timesteps))[::-1])
            for idx in pbar:
                time = torch.tensor([idx] * img.shape[0], device=device)
                denoising_steps_taken = self.num_timesteps - 1 - idx
                current_integral = self.sde_integral_term[denoising_steps_taken]

                std_history.append(img.std().item())
                timesteps_history.append(idx)

                out = self.p_sample(model=model, x=img, t=time, y=measurement, operator=operator,
                                    solve_type=solve_type, gamma = gamma, integral_value=current_integral, 
                                    mask=mask, measure_config=measure_config)
                
                img = out['sample'].detach()
                show = out['show'].detach()
                x0_t_hat = out["x0_t_hat"].detach()

                if record and idx % 100 == 0:
                # if record and idx > 990:
                    plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png"), clear_color(img))
                    plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}_show.png"), clear_color(show))
                    plt.imsave(os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}_x0_t_hat.png"), clear_color(x0_t_hat))
                    progress_images[idx] = clear_color(img)
        if record and progress_images:
            ################################ std vs timestep ################################
            fig_std, ax_std = plt.subplots(figsize=(10, 6))
            ax_std.plot(timesteps_history[::-1], std_history[::-1])
            ax_std.set_xlabel("Timestep (t)")
            ax_std.set_ylabel("Standard Deviation of img tensor")
            ax_std.set_title("Image Tensor Standard Deviation During Denoising")
            ax_std.grid(True)
            std_plot_path = os.path.join(save_root, f"progress/std_vs_timestep.png")
            plt.savefig(std_plot_path)
            plt.close(fig_std)
            ################################ save img ################################
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle('Image Generation Progress', fontsize=20)
            progress_items = reversed(list(progress_images.items()))
            for i, (idx, image) in enumerate(progress_items):
                ax = axes[i // 5, i % 5]
                ax.imshow(image)
                ax.set_title(f't = {str(idx)}')
                ax.axis('off')

            for i in range(len(progress_images), 10):
                axes[i // 5, i % 5].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            combined_image_path = os.path.join(save_root, "progress/combined_progress.png")
            plt.savefig(combined_image_path)
            plt.close(fig)
            
        return img

        
    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
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

    def p_sample_torchdiffeq(self, model, x, t, num_steps=100, solver='euler', requires_grad=False):
        current_t = t[0].item()

        if not hasattr(self, "betas_torch") or self.betas_torch.device != x.device:
            self.betas_torch = torch.from_numpy(self.betas).to(x.device, dtype=torch.float32)

        time_steps = torch.linspace(current_t, 0, num_steps+1, device=x.device)
        original_shape = x.shape

        def _ode_derivative(t_ode, x_ode):
            x_ode = x_ode.view(original_shape)
            t_discrete = torch.full((original_shape[0],), t_ode.item(), device=x.device, dtype=torch.long)
            t_discrete = torch.clamp(t_discrete, 0, self.num_timesteps - 1)
            
            with torch.no_grad():
                raw_model_output = model(x_ode, self._scale_timesteps(t_discrete))
                if raw_model_output.shape[1] == 2 * original_shape[1]:
                    model_output_mean, model_output_var = torch.split(raw_model_output, original_shape[1], dim=1)
                else:
                    model_output_mean = raw_model_output

                _, pred_xstart = self.mean_processor.get_mean_and_xstart(
                    x_ode, t_discrete, model_output_mean
                )            
            beta_t = self.betas_torch[t_discrete[0]].float()
            alpha_bar_t = self.alphas_cumprod[t_discrete[0]]

            score = (pred_xstart - x_ode) / (1.0 - alpha_bar_t + 1e-9)
            drift = -0.5 * beta_t * (x_ode + score)
            return drift.flatten(1)

        if requires_grad:
            x.requires_grad_(True)
            ode_solver = odeint_adjoint
        else:
            with torch.no_grad():
                ode_solver = odeint
        
        solution_traj = ode_solver(_ode_derivative, x.flatten(1), time_steps, method=solver, rtol=1e-5, atol=1e-5)
        x0_hat = solution_traj[-1].view(original_shape)
        
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

    def p_sample_ddim(self, model, x, t, num_steps=10):
        current_t = t[0].item()
        timesteps = np.linspace(current_t, 0, num_steps + 1, dtype=int)
        img = x.clone()

        for i in range(num_steps):
            t_now_int = timesteps[i]
            t_next_int = timesteps[i+1]
            if t_next_int == 0:
                t_next_int = -1
            time_now = torch.full((x.shape[0],), t_now_int, device=x.device, dtype=torch.long)
            with torch.no_grad():
                out = self.p_mean_variance(model, img, time_now)
                pred_xstart = out["pred_xstart"]
            eps = self._predict_eps_from_x_start(img, time_now, pred_xstart)
            if t_next_int >= 0:
                time_next = torch.full_like(time_now, t_next_int)
                alpha_bar_prev = extract_and_expand(self.alphas_cumprod, time_next, x)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=x.device)

            mean_pred = pred_xstart * torch.sqrt(alpha_bar_prev)
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps
            img = mean_pred + dir_xt
        return img
    # (x_t / sqrt(ᾱ_t) - x̂₀) / sqrt(1/ᾱ_t - 1)
    def _predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

                

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


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


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']
        # sample = x
        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample = sample + torch.exp(0.5 * out['log_variance']) * noise
            # print('out variance mean:', torch.exp(0.5 * out['log_variance']).mean().item())
            # print('out variance std:', torch.exp(0.5 * out['log_variance']).std().item())

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

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


    # (J_H(x_T_u)^T * H)x = J_H(x_T_u)^T * y
    def CG_Solve_nonlinear_infinitegamma(self, pred_xstart, y, operator, maxiter=100, tol = 1e-5):
        x0_t_hat = pred_xstart - operator.transpose(operator.forward(pred_xstart) - y)
        Hx_T = operator.forward(x0_t_hat)
        # using autodiff calculate vJp: J_H(x_T_u)^T * y
        vJp = torch.autograd.grad(
            outputs=Hx_T,
            inputs=x0_t_hat,
            grad_outputs=y,
            retain_graph=True 
        )[0]        # [1,3,256,256]

        # define A = J_H^T * H and calcualte result of A*v
        def apply_A(v_tensor):     # [1,3,256,256]
            v_tensor.requires_grad_(True)
            Hv = operator.forward(v_tensor)
            x_T_hat = operator.forward(x0_t_hat)
            result = torch.autograd.grad(
                outputs=x_T_hat,
                inputs=x0_t_hat,
                grad_outputs=Hv,
                retain_graph=True
            )[0]
            v_tensor.requires_grad_(False)
            return result

        # using CG to solve A*x = b
        b_numpy = vJp.detach().cpu().numpy().flatten()

        def matvec_for_scipy(v_flat_numpy):
            v_tensor = torch.from_numpy(v_flat_numpy).view(pred_xstart.shape).float().to(pred_xstart.device)
            Av_tensor = apply_A(v_tensor)
            return Av_tensor.detach().cpu().numpy().flatten()

        n_high_res = np.prod(pred_xstart.shape)
        A_operator = LinearOperator(shape=(n_high_res, n_high_res), matvec=matvec_for_scipy)
        x_solution_flat, info = cg(A_operator, b_numpy, maxiter=maxiter, tol=tol) # maxiter 可调
        if info != 0:
            print(f"Warning: Conjugate Gradient did not converge (info={info}). The result might be inaccurate.")
        x_solution_tensor = torch.from_numpy(x_solution_flat).view(pred_xstart.shape)  # [1,3,256,256]

        return x_solution_tensor.to(y.device)

    
    def CG_Solve_nonlinear_infinitegamma_deltax(self, pred_xstart, y, operator, maxiter=100, tol = 1e-5):
        device = pred_xstart.device
        x0_for_grad = pred_xstart.detach().clone().requires_grad_()
        Hx0 = operator.forward(x0_for_grad)
        residual = y - Hx0

        b_prime_tensor = torch.autograd.grad(
            outputs=Hx0,
            inputs=x0_for_grad,
            grad_outputs=residual,
            retain_graph=True
        )[0]

        def apply_A_prime(delta_x_tensor):
            delta_x_tensor = delta_x_tensor.to(device)
            J_delta_x = jvp(
                lambda x: operator.forward(x),
                (x0_for_grad,),
                (delta_x_tensor,)
            )[1]
            JT_J_delta_x = torch.autograd.grad(
                outputs=Hx0,
                inputs=x0_for_grad,
                grad_outputs=J_delta_x,
                retain_graph=True
            )[0]
            return JT_J_delta_x

        b_prime_numpy = b_prime_tensor.detach().cpu().numpy().flatten()
        def matvec_for_scipy(v_flat_numpy):
            v_tensor = torch.from_numpy(v_flat_numpy).view(pred_xstart.shape).float()
            Av_tensor = apply_A_prime(v_tensor)
            return Av_tensor.detach().cpu().numpy().flatten()
        
        n_high_res = np.prod(pred_xstart.shape)
        A_prime_operator = LinearOperator(shape=(n_high_res, n_high_res), matvec=matvec_for_scipy)

        delta_x_flat, info = cg(A_prime_operator, b_prime_numpy, maxiter=maxiter, tol=tol)
        if info != 0:
            print(f"Warning: Conjugate Gradient did not converge (info={info}). The result might be inaccurate.")
        delta_x_tensor = torch.from_numpy(delta_x_flat).view(pred_xstart.shape).to(device)

        return pred_xstart + delta_x_tensor


    def Solve_with_Gradient_Descent(self, pred_xstart, y, operator, num_iterations=500, lr=0.1):
        device = pred_xstart.device
        x0_for_grad = pred_xstart.detach().clone().requires_grad_()
        Hx0 = operator.forward(x0_for_grad)
        b_tensor = torch.autograd.grad(
            outputs=Hx0,
            inputs=x0_for_grad,
            grad_outputs=y,
            retain_graph=False
        )[0]

        x0_for_grad_A = pred_xstart.detach().clone().requires_grad_()
        Hx0_A = operator.forward(x0_for_grad_A)
        def apply_A(v_tensor):
            v_tensor = v_tensor.to(device)
            Hv = operator.forward(v_tensor)
            # J_H^T * (H*v)
            JTHv = torch.autograd.grad(
                outputs=Hx0_A,
                inputs=x0_for_grad_A,
                grad_outputs=Hv,
                retain_graph=True,
                create_graph=True
            )[0]
            return JTHv

        x = pred_xstart.clone()
        x_best = x.clone()
        residual_norm_best = float('inf')
        for i in range(num_iterations):
            Ax = apply_A(x)
            residual = b_tensor - Ax
            residual_norm = torch.linalg.norm(residual)

            if residual_norm < 1e-5:
                # print(f"Convergence reached at iteration {i+1}. Final Residual Norm: {residual_norm.item():.6f}")
                x_best = x.clone()
                break
            if residual_norm < residual_norm_best:
                residual_norm_best = residual_norm.item()
                x_best = x.clone()
            else:
                lr /= 10
                x = x_best.clone()
                # print(f"Iteration {i+1}/{num_iterations}, Residual Norm: {residual_norm.item():.6f}, LR: {lr}")

                if lr < 1e-8:
                    break

            x = x + lr * residual

        return x_best


    #  [I + c*J^T*H]x = J^T*(k*H*x0 - y)
    def CG_Solve_nonlinear_finitegamma(self, pred_xstart, y, operator, gamma, task, maxiter=100, tol=1e-5):
        device = pred_xstart.device
        # c = γ * (1 - exp(-Integral(g^2)))
        c_scalar = gamma * (1.0 - np.exp(-self.total_g2_integral))  
        # k = exp(-1/2 * Integral(g^2))
        k_scalar = np.exp(-0.5 * self.total_g2_integral)

        # calculate right vector J^T(k*H(x0) - y) ---
        def compute_rhs_b(vector):
            x0_for_grad = pred_xstart.detach().clone().requires_grad_()
            Hx0_for_grad = operator.forward(x0_for_grad)

            rhs_b = torch.autograd.grad(
                outputs=Hx0_for_grad,
                inputs=x0_for_grad,
                grad_outputs=vector,
                retain_graph=False
            )[0]
            return rhs_b

        # A*v = v + c * J^T*H*v where A = [I + c*J^T*H]
        def apply_operator_A(v_tensor):
            v_tensor = v_tensor.to(device)
            v_tensor.requires_grad_(True)
            
            # calculate J^T*H*v
            Hv = operator.forward(v_tensor)
            x0_hat = pred_xstart.detach().clone().requires_grad_()
            Hx0_for_grad = operator.forward(x0_hat)

            JTHv = torch.autograd.grad(
                outputs=Hx0_for_grad,
                inputs=x0_hat,
                grad_outputs=Hv,
                retain_graph=True
            )[0]
            v_tensor.requires_grad_(False)

            return v_tensor + c_scalar * JTHv

        def matvec_for_scipy(v_flat_numpy):
            v_tensor = torch.from_numpy(v_flat_numpy).view(pred_xstart.shape).float().to(pred_xstart.device)
            Av_tensor = apply_operator_A(v_tensor)
            return Av_tensor.detach().cpu().numpy().flatten()

        n_dims = np.prod(pred_xstart.shape)
        A_linear_op = LinearOperator(shape=(n_dims, n_dims), matvec=matvec_for_scipy)
        
        Hx0 = operator.forward(self.x_start)
        b1_tensor = compute_rhs_b(k_scalar * Hx0 - y)
        b1_numpy = b1_tensor.detach().cpu().numpy().flatten()
        x1_solution_flat, info1 = cg(A_linear_op, b1_numpy, maxiter=maxiter, tol=tol)
        if info1 != 0:
            print(f"Warning: Conjugate Gradient did not converge (info1={info1}).")
        x1_solution = torch.from_numpy(x1_solution_flat).view(pred_xstart.shape).to(device)

        if task in ['inpainting','inpainting_soc','super_resolution_inpainting_soc','super_resolution_inpainting']:
            b2_tensor = compute_rhs_b(operator.forward(pred_xstart))
            b2_numpy =  b2_tensor.detach().cpu().numpy().flatten()
            x2_solution_flat, info2 = cg(A_linear_op, b2_numpy, maxiter=50, tol=1e-5)
            if info2 != 0:
                print(f"Warning: Conjugate Gradient did not converge (info2={info2}).")
            x2_solution = torch.from_numpy(x2_solution_flat).view(pred_xstart.shape).to(device)
            return x1_solution, x2_solution
        else:
            return x1_solution

    
    def CG_Solve_nonlinear_finitegamma_deltax(self, pred_xstart, y, operator, gamma, maxiter=100, tol=1e-5):
        device = pred_xstart.device
        dtype = pred_xstart.dtype
        c_scalar = gamma * (1.0 - np.exp(-self.total_g2_integral))  
        k_scalar = np.exp(-0.5 * self.total_g2_integral)

        x_u_for_grad = pred_xstart.detach().clone().requires_grad_(True)
        Hx_u = operator.forward(x_u_for_grad)

        with torch.no_grad():
            Hx0 = operator.forward(self.x_start)
        
        # calculate c * J_H(x_T^u) * x_T^u
        _, c_J_xu = jvp(
            lambda x: operator.forward(x),
            (x_u_for_grad,),
            (pred_xstart,)
        )
        
        vec_inside = k_scalar * Hx0 - y - c_scalar * Hx_u.detach() + c_scalar * c_J_xu
        b_prime_tensor = torch.autograd.grad(
            outputs=Hx_u,
            inputs=x_u_for_grad,
            grad_outputs=vec_inside,
            retain_graph=True
        )[0].detach()

        # A'(v) = (I + c * J_H(x_T^u)^T * J_H(x_T^u)) @ v
        def apply_A_prime(v_tensor):
            v_tensor = v_tensor.to(device, dtype)
            J_v = jvp(
                lambda var: operator.forward(var), (x_u_for_grad,), (v_tensor,)
            )[1]
            JT_J_v = torch.autograd.grad(
                outputs=Hx_u, inputs=x_u_for_grad, grad_outputs=J_v, retain_graph=True
            )[0]
            return v_tensor + c_scalar * JT_J_v

        b_prime_numpy = b_prime_tensor.cpu().numpy().flatten()
        def matvec_for_scipy(v_numpy):
            v_tensor = torch.from_numpy(v_numpy).view(pred_xstart.shape).to(device, dtype)
            Av_tensor = apply_A_prime(v_tensor)
            return Av_tensor.detach().cpu().numpy().flatten()
        
        n_elements = np.prod(pred_xstart.shape)
        A_prime_operator = LinearOperator(shape=(n_elements, n_elements), matvec=matvec_for_scipy)
        solution_x_flat, info = cg(A_prime_operator, b_prime_numpy, maxiter=maxiter, tol=tol)
        if info != 0:
            print(f"Warning: Conjugate Gradient did not converge (info={info}).")
        solution_x = torch.from_numpy(solution_x_flat).view(pred_xstart.shape).to(device, dtype)
        
        return -gamma * solution_x.detach()

    
    def Solve_with_MCMC(self, model, xt, t, pred_xstart, operator, measurement):
        time_step = t.item()
        ratio = ((self.num_timesteps - 1 - time_step) / (self.num_timesteps - 1)) if self.num_timesteps > 1 else 1.0
        x0_t_hat = self.mcmc_sampler.sample(xt=xt, model=model, x0hat=pred_xstart, operator=operator, 
                                       measurement=measurement, sigma=self.sigma_steps[time_step], ratio=ratio)

        return x0_t_hat

    def Solve_with_MCMC_finitegamma(self, model, xt, t, pred_xstart, operator, measurement, gamma):
        time_step = t.item()
        ratio = ((self.num_timesteps - 1 - time_step) / (self.num_timesteps - 1)) if self.num_timesteps > 1 else 1.0
        c_scalar = gamma * (1.0 - np.exp(-self.total_g2_integral))
        k_scalar = np.exp(-0.5 * self.total_g2_integral)

        x_solution = self.mcmc_sampler.sample_finitegamma(xt=xt, model=model, pred_xstart=pred_xstart, operator=operator, measurement=measurement, sigma=self.sigma_steps[time_step], ratio=ratio, c_scalar=c_scalar, k_scalar=k_scalar, gamma=gamma, x_start=self.x_start)
        
        return -x_solution


    def Solve_with_MCMC_finitegamma_energy(self, model, xt, t, pred_xstart, operator, measurement, gamma):
        time_step = t.item()
        # ratio = ((self.num_timesteps - 1 - time_step) / (self.num_timesteps - 1)) if self.num_timesteps > 1 else 1.0
        lr = 1
        c_scalar = gamma * (1.0 - np.exp(-self.total_g2_integral))
        k_scalar = np.exp(-0.5 * self.total_g2_integral)

        x_solution = self.mcmc_sampler.sample_langevin_true_energy(xt=xt, model=model, pred_xstart=pred_xstart, operator=operator, measurement=measurement, sigma=self.sigma_steps[time_step], lr=lr, c_scalar=c_scalar, k_scalar=k_scalar, x_start=self.x_start, gamma=gamma)

        return -gamma * x_solution


    def p_sample(self, model, x, t, y, operator, solve_type, gamma, integral_value, mask=None, measure_config=None):
        # x_0 prediction
        # out = self.p_mean 
        out = self.p_mean_variance(model, x, t)
        pred_xstart = out['pred_xstart']

        g2_t = torch.full((x.shape[0],), self.g2_schedule[t], device=x.device)
        g2_t = g2_t.view(-1, 1, 1, 1)
        # exp(1/2 * integral from 0 to t_sde)
        inv_alpha_t_sde = integral_value
        dt = (1/ self.num_timesteps)

        # drift term
        uncond_drift = -0.5 * g2_t * out['mean']
        if solve_type == 'linear':
            if measure_config['operator']['name'] in ['inpainting_soc','super_resolution_inpainting_soc']:
                if measure_config['mask_opt']['mask_type'] == 'random':
                    pred_xstart = pred_xstart - operator.transpose(operator.forward(pred_xstart)) + operator.transpose(y)
                else:
                    pred_xstart = pred_xstart - operator.transpose(operator.forward(pred_xstart)) + operator.transpose(y)*mask + pred_xstart*(1-mask)
            else:
                pred_xstart = pred_xstart - operator.transpose(operator.forward(pred_xstart)) + operator.transpose(y)
                # x0_t_hat = operator.transpose(y)

            sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
            x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
            x0_t_hat = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=5, solver='euler')

            # x_t_obs_exp = self.q_sample(x0_t_hat, t)
            # cond_drift = g2_t * (inv_alpha_t_sde * self.alpha_T_sde * self.x_start - x_t_obs_exp / self.alpha_T_sde) / self.sde_denominator
            cond_numerator = self.alpha_T_sde * self.x_start - x0_t_hat
            cond_drift = g2_t * inv_alpha_t_sde * (cond_numerator / self.sde_denominator)

            if measure_config['operator']['name'] == 'inpainting_soc':
                mk_tp = measure_config['mask_opt']['mask_type']
                drift = uncond_drift + cond_drift * measure_config['scale']['linear'][mk_tp]
            else:
                # drift = uncond_drift + cond_drift * measure_config['scale']['linear']
                # drift = (uncond_drift + cond_drift) * self.snr[t[0].item()] * 1.5
                drift = uncond_drift + cond_drift
                # drift = uncond_drift + cond_drift * self.scale_steps_list[t[0].item()]
        elif solve_type == 'nonlinear':
            #############################################
            if t[0].item() != 0:
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                pred_xstart = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=5, solver='euler')
                # pred_xstart = self.p_sample_torchdiffeq(model, x, t, num_steps=25, solver='euler')
                # pred_xstart = self.p_sample_ddim(model, x, t, num_steps=25)
            x0_t_hat = self.Solve_with_MCMC(model, x, t, pred_xstart, operator, y)
            # x0_t_hat = self.CG_Solve_nonlinear_infinitegamma_deltax(pred_xstart, y, operator, maxiter=300, tol = 1e-2)
            #############################################
            # if measure_config['operator']['name'] in ['inpainting_soc']:
            #     x0_t_hat = y*mask + x0_t_hat*(1-mask)
                # noisy_H_dagger_y = self.q_sample(x_start=x0_t_hat, t=t)
                # x0_t_hat = self.p_mean_variance(model, noisy_H_dagger_y, t)['pred_xstart']
            cond_numerator = self.alpha_T_sde * self.x_start - x0_t_hat
            cond_drift = g2_t * inv_alpha_t_sde * (cond_numerator / self.sde_denominator)

            if measure_config['operator']['name'] == 'inpainting_soc':
                mk_tp = measure_config['mask_opt']['mask_type']
                drift = uncond_drift + cond_drift * measure_config['scale']['nonlinear'][mk_tp]
            else:
                drift = uncond_drift + cond_drift * measure_config['scale']['nonlinear']
        elif solve_type == 'nonlinear_finitegamma':
            #############################################
            if t[0].item() != 0:
                sigma_PFODE = self.sigma_steps_list.to(x.device)[self.num_timesteps-1-t].item()
                x_edm_start = pred_xstart + torch.randn_like(pred_xstart) * sigma_PFODE
                pred_xstart = self.p_sample_torchdiffeq_edm(self.edm_model, x_edm_start, t, sigma_PFODE, num_steps=5, solver='euler')
            #     pred_xstart = self.p_sample_torchdiffeq(model, x, t, num_steps=5, solver='euler')
            if measure_config['operator']['name'] in ['inpainting_soc']:
                x_hat = x * mask + pred_xstart * (1-mask)
                # x0_t_hat = self.Solve_with_MCMC_finitegamma(model, x_hat, t, pred_xstart, operator, y, gamma=gamma)
                x0_t_hat = self.CG_Solve_nonlinear_finitegamma_deltax(x_hat, y, operator, gamma, maxiter=300, tol=1e-5)
                x0_t_hat = x0_t_hat * mask + pred_xstart * (1-mask)
            else:
                # x0_t_hat = self.CG_Solve_nonlinear_finitegamma(pred_xstart, y, operator, gamma, measure_config['operator']['name'], maxiter=300, tol=1e-5)
                # x0_t_hat = self.CG_Solve_nonlinear_finitegamma_deltax(pred_xstart, y, operator, gamma, maxiter=300, tol=1e-3)
                x0_t_hat = self.Solve_with_MCMC_finitegamma(model, x, t, pred_xstart, operator, y, gamma=gamma)
                # x0_t_hat = self.Solve_with_MCMC_finitegamma_energy(model, x, t, pred_xstart, operator, y, gamma=gamma)
            #############################################
                
            e_factor = integral_value * np.exp(-0.5 * self.total_g2_integral)
            cond_drift = g2_t * e_factor * x0_t_hat
            if measure_config['operator']['name'] == 'inpainting_soc':
                mk_tp = measure_config['mask_opt']['mask_type']
                drift = uncond_drift + cond_drift * measure_config['scale']['nonlinear_finitegamma'][mk_tp]
            else:
                drift = uncond_drift + cond_drift * measure_config['scale']['nonlinear_finitegamma']
        
        # diffusion term
        noise = torch.randn_like(x)
        diffusion = torch.sqrt(g2_t) * noise
        # print('aaaa:',torch.sqrt(g2_t)* math.sqrt(dt))
        # print('log_variance:',torch.exp(0.5 * out['log_variance'])[0,0,0,0])
        ###################################### Euler-Maruyama undate ######################################
        sample = out['mean']
        # sample = x
        if t[0].item() == 0:
            sample = sample + drift * dt
        elif t > torch.tensor([0] * x.shape[0], device=x.device):
            sample = sample + drift * dt + diffusion * math.sqrt(dt)
            # sample = sample + drift * dt + torch.exp(0.5 * out['log_variance']) * noise
            # sample = sample + drift * dt

        

        return {"sample": sample, "pred_xstart": pred_xstart, "show": pred_xstart, "x0_t_hat": x0_t_hat}
    

@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t, eta=0.0):
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

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
