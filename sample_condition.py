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
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, clear_color_phase_retrieval
from util.logger import get_logger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance


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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, choices=['ffhq', 'imagenet'], default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop_sde, model=model)

    # Working directory
    save_dir = os.path.join(args.save_dir, f'{args.dataset}')
    if measure_config['operator']['name'] == 'inpainting_soc':
        if measure_config['mask_opt']['mask_type'] == 'box':
            out_path = os.path.join(save_dir, 'inpainting_box_soc')
        else:
            out_path = os.path.join(save_dir, 'inpainting_random_soc')
    else:
        out_path = os.path.join(save_dir, measure_config['operator']['name'])

    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'recon_nonlinear', 'recon_nonlinear_finitegamma']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = dict(task_config['data'])
    if args.dataset is not None:
        data_config['root'] = data_config['root'][args.dataset]
        data_config['name'] = args.dataset
    dataset = get_dataset(**data_config)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if 'inpainting' in measure_config['operator']['name'].split('_'):
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    if args.solve_type == 'linear':
        results_dir = os.path.join(out_path, 'recon')
    elif args.solve_type == 'nonlinear':
        results_dir = os.path.join(out_path, 'recon_nonlinear')
    elif args.solve_type == 'nonlinear_finitegamma':
        results_dir = os.path.join(out_path, 'recon_nonlinear_finitegamma',str(args.gamma))
    print('results_dir:',results_dir)
    os.makedirs(results_dir, exist_ok=True)
    output_filename = os.path.join(results_dir, "results.txt")
    with open(output_filename, "a") as f:
        f.write("Image_Index,TASK,Method,GAMMA,PSNR,SSIM,LPIPS\n")
    
    # Do Inference
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    psnr_all, ssim_all, lpips_all = [], [], []
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        if measure_config['operator']['name'] in ['inpainting','inpainting_soc','super_resolution_inpainting_soc','super_resolution_inpainting']:
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            sample_fn = partial(sample_fn, mask=mask)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)
        
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(img_idx=i, x_start=x_start, edm_sigma_max=args.edm_sigma_max,measurement=y_n, operator = operator, record=True, save_root=out_path, solve_type=args.solve_type, gamma=args.gamma, measure_config=measure_config, model_config=model_config, results_dir=results_dir)

        ref_numpy = (ref_img.detach().cpu().squeeze().numpy() + 1) / 2
        ref_numpy = np.transpose(ref_numpy, (1, 2, 0))

        sample_numpy = (sample.detach().cpu().squeeze().numpy() + 1) / 2
        sample_numpy = np.transpose(sample_numpy, (1, 2, 0))
        # calculate fid
        real_u8 = (np.clip(ref_numpy,   0.0, 1.0) * 255.0).round().astype(np.uint8)
        fake_u8 = (np.clip(sample_numpy, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        real_u8_t = torch.from_numpy(real_u8).permute(2, 0, 1).unsqueeze(0).to(device)
        fake_u8_t = torch.from_numpy(fake_u8).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(real_u8_t, real=True)
        fid.update(fake_u8_t, real=False)
        # calculate psnr
        psnr = peak_signal_noise_ratio(ref_numpy, sample_numpy, data_range=1.0)
        # calculate ssim
        ssim = structural_similarity(ref_numpy, sample_numpy, data_range=1.0, multichannel=True, channel_axis=-1)
        # calculate lpips
        rec_img_torch = torch.from_numpy(sample_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        gt_img_torch = torch.from_numpy(ref_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        rec_img_torch = rec_img_torch * 2 - 1
        gt_img_torch = gt_img_torch * 2 - 1
        lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()

        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips_alex)

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

    psnr_all = np.asarray(psnr_all, dtype=np.float64)
    ssim_all = np.asarray(ssim_all, dtype=np.float64)
    lpips_all = np.asarray(lpips_all, dtype=np.float64)
    psnr_mean, psnr_std, psnr_var = float(psnr_all.mean()), float(psnr_all.std()), float(psnr_all.var(ddof=0))
    ssim_mean, ssim_std, ssim_var = float(ssim_all.mean()), float(ssim_all.std()), float(ssim_all.var(ddof=0))
    lpips_mean, lpips_std, lpips_var = float(lpips_all.mean()), float(lpips_all.std()), float(lpips_all.var(ddof=0))
    fid_value = float(fid.compute().item())

    task_name = measure_config['operator']['name']
    method = getattr(args, 'solve_type', 'NA')
    gamma  = getattr(args, 'gamma', 'NA')
    summary_path = os.path.join(results_dir, "summary_metrics.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as fsum:
        fsum.write(
            f"TASK={task_name}, METHOD={method}, GAMMA={gamma}, N={len(psnr_all)}, "
            f"PSNR_mean={psnr_mean:.6f}, PSNR_std={psnr_std:.6f}, PSNR_var={psnr_var:.6f}, "
            f"SSIM_mean={ssim_mean:.6f}, SSIM_std={ssim_std:.6f}, SSIM_var={ssim_var:.6f}, "
            f"LPIPS_mean={lpips_mean:.6f}, LPIPS_std={lpips_std:.6f}, LPIPS_var={lpips_var:.6f}, "
            f"FID={fid_value:.6f}\n"
        )

if __name__ == '__main__':
    main()