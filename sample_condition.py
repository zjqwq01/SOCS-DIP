from functools import partial
import os
import argparse
import yaml
import lpips
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import get_model
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, clear_color_phase_retrieval
from util.logger import get_logger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--solve_type',type=str,default='linear',choices=['linear','nonlinear','nonlinear_finitegamma'])
    parser.add_argument('--edm_sigma_max', type=int, default=100)
    parser.add_argument('--time_travel', type=bool, default=False, help='Whether to use time travel in sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
   
    # logger
    logger = get_logger()
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    if 'soc' in measure_config['operator']['name'].split('_'):
        sample_fn = partial(sampler.p_sample_loop_sde, model=model)
    else:
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    if measure_config['operator']['name'] == 'inpainting_soc':
        if measure_config['mask_opt']['mask_type'] == 'box':
            out_path = os.path.join(args.save_dir, 'inpainting_box_soc')
        else:
            out_path = os.path.join(args.save_dir, 'inpainting_random_soc')
    else:
        out_path = os.path.join(args.save_dir, measure_config['operator']['name'])

    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'recon_nonlinear', 'recon_nonlinear_finitegamma']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] in ['inpainting','inpainting_soc','super_resolution_inpainting_soc','super_resolution_inpainting']:
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    if args.solve_type == 'linear':
        results_dir = os.path.join(out_path, 'recon')
    elif args.solve_type == 'nonlinear':
        results_dir = os.path.join(out_path, 'recon_nonlinear')
    elif args.solve_type == 'nonlinear_finitegamma':
        results_dir = os.path.join(out_path, 'recon_nonlinear_finitegamma',str(args.gamma))
    os.makedirs(results_dir, exist_ok=True)
    output_filename = os.path.join(results_dir, "results.txt")
    with open(output_filename, "w") as f:
        f.write("Image_Index,TASK,Method,GAMMA,PSNR,SSIM,LPIPS\n")
    
    # Do Inference
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)        # [1,3,256,256]

        # Exception) In case of inpainting,
        if measure_config['operator']['name'] in ['inpainting','inpainting_soc','super_resolution_inpainting_soc','super_resolution_inpainting']:
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            if 'soc' in measure_config['operator']['name'].split('_'):
                sample_fn = partial(sample_fn, mask=mask)
            else:
                sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)
        
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        if 'soc' in measure_config['operator']['name'].split('_'):
            if measure_config['operator']['name'] in ['phase_retrieval_soc']:
                x_start = x_start * args.edm_sigma_max
            sample = sample_fn(img_idx=i, x_start=x_start, edm_sigma_max=args.edm_sigma_max,measurement=y_n, operator = operator, record=True, save_root=out_path, solve_type=args.solve_type, gamma=args.gamma, time_travel=args.time_travel, measure_config=measure_config, model_config=model_config, results_dir=results_dir)
        else:
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        ref_numpy = (ref_img.detach().cpu().squeeze().numpy() + 1) / 2
        ref_numpy = np.transpose(ref_numpy, (1, 2, 0))

        sample_numpy = (sample.detach().cpu().squeeze().numpy() + 1) / 2
        sample_numpy = np.transpose(sample_numpy, (1, 2, 0))
        # calculate psnr
        psnr = peak_signal_noise_ratio(ref_numpy, sample_numpy)
        ssim = structural_similarity(ref_numpy, sample_numpy, data_range=1.0, multichannel=True, channel_axis=-1)
        # calculate lpips
        rec_img_torch = torch.from_numpy(sample_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        gt_img_torch = torch.from_numpy(ref_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        rec_img_torch = rec_img_torch * 2 - 1
        gt_img_torch = gt_img_torch * 2 - 1
        lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()

        print('TASK:{} Method:{} GAMMA:{} PSNR: {}, SSIM: {}, LPIPS: {}'.format(measure_config['operator']['name'], args.solve_type, args.gamma, psnr, ssim, lpips_alex))
        output_string = '{},{},{},{},{},{},{}'.format(i, measure_config['operator']['name'], args.solve_type, args.gamma, psnr,ssim, lpips_alex)
        with open(output_filename, "a") as f:
            f.write(output_string + "\n")

        if measure_config['operator']['name'] in ['phase_retrieval', 'phase_retrieval_soc']:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color_phase_retrieval(y_n))
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(results_dir, fname), clear_color(sample))

if __name__ == '__main__':
    main()


# #super_resolution
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/super_resolution_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/super_resolution_config.yaml 

# #colorization
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/colorization_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/colorization_config.yaml

# #inpainting
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/inpainting_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7 --time_travel True
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/inpainting_config.yaml

# #super_resolution_inpainting
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/super_resolution_inpainting_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/super_resolution_inpainting_config.yaml

# #motion_deblur
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/motion_deblur_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/motion_deblur_config.yaml

# #nonlinear_deblur
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/nonlinear_deblur_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/nonlinear_deblur_config.yaml

# #phase_retrieval
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/phase_retrieval_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/phase_retrieval_config.yaml

# high_dynamic_range
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_soc_config.yaml --task_config=configs/high_dynamic_range_soc_config.yaml --solve_type nonlinear_finitegamma --gamma 1e7
# python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/high_dynamic_range_config.yaml
