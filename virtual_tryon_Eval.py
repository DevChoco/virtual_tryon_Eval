from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
from PIL import Image
import numpy as np
import torch
import os
from tqdm import tqdm
import lpips
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

original_dir = "original_images"
synthesized_dir = "synthesized_images"
resize_size = (256, 256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

inception.Mixed_7c.register_forward_hook(get_activation('pool3'))

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

lpips_model = lpips.LPIPS(net='vgg').to(device)

def extract_feature(img_list):
    feats = []
    with torch.no_grad():
        for img in tqdm(img_list, desc="Extracting features (Inception)"):
            x = transform(img).unsqueeze(0).to(device)
            _ = inception(x) 
            feat = activation['pool3']
            feat = adaptive_avg_pool2d(feat, output_size=(1, 1)).squeeze().cpu().numpy()
            feats.append(feat)
    feats = np.stack(feats, axis=0)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)


original_images = []
synthesized_images = []
lpips_scores = []
psnr_scores = []
ssim_scores = []

for file_name in os.listdir(original_dir):
    original_path = os.path.join(original_dir, file_name)
    synthesized_path = os.path.join(synthesized_dir, f"gen_{file_name[:8]}_{file_name}")

    try:
        orig_img = Image.open(original_path).convert("RGB")
        synth_img = Image.open(synthesized_path).convert("RGB")
    except Exception as e:
        print(f"[로드 실패] {file_name}: {e}")
        continue

    original_images.append(orig_img)
    synthesized_images.append(synth_img)

    lpips_input1 = TF.resize(orig_img, resize_size)
    lpips_input2 = TF.resize(synth_img, resize_size)
    lpips_input1 = TF.to_tensor(lpips_input1).unsqueeze(0).to(device)
    lpips_input2 = TF.to_tensor(lpips_input2).unsqueeze(0).to(device)
    with torch.no_grad():
        d = lpips_model(lpips_input1, lpips_input2).item()
        lpips_scores.append(d)

    np_orig = np.array(orig_img.resize(resize_size)).astype(np.float32)
    np_synth = np.array(synth_img.resize(resize_size)).astype(np.float32)

    psnr_val = psnr(np_orig, np_synth, data_range=255)
    ssim_val = ssim(np_orig, np_synth, win_size=11, channel_axis=-1, data_range=255)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

if len(original_images) == 0 or len(synthesized_images) == 0:
    raise RuntimeError("오리지널 또는 합성 이미지가 없습니다.")

mu_orig, sigma_orig = extract_feature(original_images)
mu_synth, sigma_synth = extract_feature(synthesized_images)
fid_score = calculate_fid(mu_orig, sigma_orig, mu_synth, sigma_synth)


print("이미지 품질 평가 결과:")
print(f"FID Score   : {fid_score:.4f} (↓ 낮을수록 좋음)")
print(f"LPIPS Score : {np.mean(lpips_scores):.4f} (↓ 낮을수록 좋음)")
print(f"PSNR        : {np.mean(psnr_scores):.4f} dB (↑ 높을수록 좋음)")
print(f"SSIM        : {np.mean(ssim_scores):.4f} (↑ 높을수록 좋음)")
