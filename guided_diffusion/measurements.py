'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import torch.nn as nn
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import numpy as np
from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, ifft2_m, fft2c_new, ifft2c_new


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution_soc')
class SuperResolutionSocOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.down_sample = torch.nn.AdaptiveAvgPool2d((in_shape[-1]//scale_factor,in_shape[-1]//scale_factor))
        def MeanUpsample(z, scale_factor):
            n, c, h, w = z.shape
            out = torch.zeros(n, c, h, scale_factor, w, scale_factor).to(z.device) + z.view(n,c,h,1,w,1)
            out = out.view(n, c, scale_factor*h, scale_factor*w)
            return out
        self.pseudo_inverse = lambda z: MeanUpsample(z, scale_factor)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.pseudo_inverse(data)


@register_operator(name='colorization_soc')
class ColorizationSocOperator(LinearOperator):
    def __init__(self, in_shape, device):
        self.device = device
        def color2gray(x):
            coef=1/3
            x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
            return x.repeat(1,3,1,1)
        def gray2color(x):
            x = x[:,0,:,:]
            coef=1/3
            base = coef**2 + coef**2 + coef**2
            return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)
        self.c2g = lambda x: color2gray(x)
        self.g2c = lambda x: gray2color(x)

    def forward(self, data, **kwargs):
        return self.c2g(data)

    def transpose(self, data, **kwargs):
        return self.g2c(data)


@register_operator(name='motion_blur_soc')
class MotionBlurSocOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
        self.conv.requires_grad_(False)

        # self.transpose_conv = nn.ConvTranspose2d(in_channels=3, out_channels=3, 
        #                                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, 
        #                                          groups=3, bias=False).to(device)
    
    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        # return self.transpose_conv(data)
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur_soc')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting_soc')
class InpaintingSocOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
        self.mask = None
    
    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', self.mask)
        if mask is None:
            raise ValueError("Require mask for InpaintingSocOperator")
        self.mask = mask 

        return data * self.mask.to(self.device)

    def transpose(self, data, **kwargs):
        try:
            return data * self.mask.to(self.device)
        except:
            raise ValueError("Require mask")


@register_operator(name='super_resolution_inpainting_soc')
class SuperResolutionInpaintingSocOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.down_sample = torch.nn.AdaptiveAvgPool2d((in_shape[-1]//scale_factor,in_shape[-1]//scale_factor))
        def MeanUpsample(z, scale_factor):
            n, c, h, w = z.shape
            out = torch.zeros(n, c, h, scale_factor, w, scale_factor).to(z.device) + z.view(n,c,h,1,w,1)
            out = out.view(n, c, scale_factor*h, scale_factor*w)
            return out
        self.pseudo_inverse = lambda z: MeanUpsample(z, scale_factor)
        self.mask = None

    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', self.mask)
        if mask is None:
            raise ValueError("Require mask for SuperResolutionInpaintingSocOperator")
        self.mask = mask

        A1 = data * self.mask.to(self.device)
        A2 = self.down_sample(A1)
        
        return A2

    def transpose(self, data, **kwargs):
        A2p = self.pseudo_inverse(data)
        try:
            A1p = A2p * self.mask.to(self.device)
        except:
            raise ValueError("Require mask")
        
        return A1p



class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


@register_operator(name='phase_retrieval_soc')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        data = data * 0.5 + 0.5
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def transpose(self, data, **kwargs):
        img_padded = ifft2_m(data).abs()
        _B, _C, H, W = img_padded.shape
        cropped_img = img_padded[..., self.pad : H - self.pad, self.pad : W - self.pad]
        return cropped_img


@register_operator(name='nonlinear_blur_soc')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)
        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

    def transpose(self, data, **kwargs):
        return data


@register_operator(name='high_dynamic_range_soc')
class HighDynamicRangeOperator(NonLinearOperator):
    def __init__(self, scale, device):
        self.device = device
        self.scale = scale

    def forward(self, data, **kwargs):
        return torch.clip((data * self.scale), -1, 1)

    def transpose(self, data, **kwargs):
        return data



# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)