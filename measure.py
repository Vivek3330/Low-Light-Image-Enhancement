import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 🔥 FORCE CPU

import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform


device = torch.device("cpu")   # 🔥 GLOBAL DEVICE


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)

    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([ssim(img1[:,:,i], img2[:,:,i]) for i in range(3)])
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))

    raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    return 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))


def metrics(im_dir, label_dir, use_GT_mean):
    avg_psnr, avg_ssim, avg_lpips = 0, 0, 0
    n = 0

    # 🔥 LPIPS CPU
    loss_fn = lpips.LPIPS(net='alex').to(device)

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        im1 = Image.open(item).convert('RGB')

        name = os.path.basename(item)
        im2 = Image.open(label_dir + name).convert('RGB')

        (h, w) = im2.size
        im1 = im1.resize((h, w))

        im1 = np.array(im1)
        im2 = np.array(im2)

        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)

        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        # 🔥 REMOVE CUDA
        ex_p0 = lpips.im2tensor(im1).to(device)
        ex_ref = lpips.im2tensor(im2).to(device)

        score_lpips = loss_fn.forward(ex_ref, ex_p0)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()

    avg_psnr /= n
    avg_ssim /= n
    avg_lpips /= n

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':

    mea_parser = argparse.ArgumentParser(description='Measure')
    mea_parser.add_argument('--use_GT_mean', action='store_true')
    mea_parser.add_argument('--lol', action='store_true')
    mea_parser.add_argument('--lol_v2_real', action='store_true')
    mea_parser.add_argument('--lol_v2_syn', action='store_true')
    mea_parser.add_argument('--SICE_grad', action='store_true')
    mea_parser.add_argument('--SICE_mix', action='store_true')
    mea_parser.add_argument('--fivek', action='store_true')
    mea = mea_parser.parse_args()

    if mea.lol:
        im_dir = './output/LOLv1/*.png'
        label_dir = './datasets/LOLdataset/eval15/high/'
    if mea.lol_v2_real:
        im_dir = './output/LOLv2_real/*.png'
        label_dir = './datasets/LOLv2/Real_captured/Test/Normal/'
    if mea.lol_v2_syn:
        im_dir = './output/LOLv2_syn/*.png'
        label_dir = './datasets/LOLv2/Synthetic/Test/Normal/'
    if mea.SICE_grad:
        im_dir = './output/SICE_grad/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.SICE_mix:
        im_dir = './output/SICE_mix/*.png'
        label_dir = './datasets/SICE/SICE_Reshape/'
    if mea.fivek:
        im_dir = './output/fivek/*.jpg'
        label_dir = './datasets/FiveK/test/target/'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)

    print(f"===> Avg.PSNR: {avg_psnr:.4f} dB")
    print(f"===> Avg.SSIM: {avg_ssim:.4f}")
    print(f"===> Avg.LPIPS: {avg_lpips:.4f}")