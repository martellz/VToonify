import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import math
import argparse
import torch
from torch.utils import data
from torch.nn import functional as F
from torch import autograd
from torch.nn import init
import torchvision.transforms as transforms
from model.stylegan.op_cpu import conv2d_gradfix
from model.encoder.encoders.psp_encoders import GradualStyleEncoder
from model.encoder.align_all_parallel import get_landmark
from mediapipe.python.solutions.face_detection import FaceDetection
    
def visualize(img_arr, dpi):
    plt.figure(figsize=(10,10),dpi=dpi)
    plt.imshow(((img_arr.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    plt.show()

def save_image(img, filename):
    tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    
def load_image(filename):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)   

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

            
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)    
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0) 
            
            
def load_psp_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts['n_styles'] = int(math.log(opts['output_size'], 2)) * 2 - 2
    opts = argparse.Namespace(**opts)
    psp = GradualStyleEncoder(50, 'ir_se', opts)
    psp_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    psp.load_state_dict(psp_dict)
    psp.eval()
    psp = psp.to(device)
    latent_avg = ckpt['latent_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    psp.register_forward_hook(add_latent_avg)
    return psp

def get_video_crop_parameter(filepath, predictor, padding=[200,200,200,200]):
    if type(filepath) == str:
        img = dlib.load_rgb_image(filepath)
    else:
        img = filepath
    lm = get_landmark(img, predictor)
    if lm is None:
        return None
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise
    
    scale = 64. / (np.mean(lm_eye_right[:,0])-np.mean(lm_eye_left[:,0]))
    center = ((np.mean(lm_eye_right, axis=0)+np.mean(lm_eye_left, axis=0)) / 2) * scale
    h, w = round(img.shape[0] * scale), round(img.shape[1] * scale)
    left = max(round(center[0] - padding[0]), 0) // 8 * 8
    right = min(round(center[0] + padding[1]), w) // 8 * 8
    top = max(round(center[1] - padding[2]), 0) // 8 * 8
    bottom = min(round(center[1] + padding[3]), h) // 8 * 8
    return h,w,top,bottom,left,right,scale

def get_crop_parameter_by_mediapipe(img: np.ndarray, faceDetection: FaceDetection, padding=[200, 200, 200, 200]):
    results = faceDetection.process(img)
    if results.detections:
        keypoints = results.detections[0].location_data.relative_keypoints

        lm_eye_left = np.array([keypoints[0].x * img.shape[1], keypoints[0].y * img.shape[0]])
        lm_eye_right = np.array([keypoints[1].x * img.shape[1], keypoints[1].y * img.shape[0]])
        center_before_scale = 0.5 * (lm_eye_left + lm_eye_right)
        H, W, _ = img.shape
        scale = 64. / (lm_eye_right[0] - lm_eye_left[0])
        left = max(round(center_before_scale[0] - padding[0] // scale), 0)
        right = min(round(center_before_scale[0] + padding[1] // scale), W)
        top = max(round(center_before_scale[1] - padding[2] // scale), 0)
        bottom = min(round(center_before_scale[1] + padding[3] // scale), H)
        w = int(scale * (right - left) // 8 * 8)
        h = int(scale * (bottom - top) // 8 * 8)
        return h,w,top,bottom,left,right
    else:
        return None

def creat_weight_kernel(kernel_size=(3, 3), a=0.8, b = 1.0, k=1):
    # if sigma == 0:
    #     sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size[0])
    Y = np.linspace(-k, k, kernel_size[1])
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    # kernel = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))

    distance = np.where(np.abs(x - x0) > np.abs(y - y0), np.abs(x - x0), np.abs(y - y0))    # from 0 to 1

    kernel = np.where(distance > a, (b - distance) / (b - a), 1)
    kernel = np.where(kernel < 0, 0, kernel)

    # kernel = (1 - distance) ** 1.5
    # kernel = (kernel - kernel.min()) / kernel.max()

    return kernel

def create_weight_field(mask: np.ndarray, kernel_size=(3, 3), iterations: int=10, a: float=2):
    origin = mask.copy()
    if origin.max() > 1:
        origin = origin / 255.

    temp = origin
    for _ in range(iterations):
        temp = (cv2.erode(temp, np.ones(kernel_size)) + temp) / 2
    return temp / temp.max()

def tensor2cv2(img):
    tmp = ((img.cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    return cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

# get parameters from the stylegan and mark them with their layers
def gather_params(G):
    params = dict(
        [(res, {}) for res in range(18)] + [("others", {})]
    )
    for n, p in sorted(list(G.named_buffers()) + list(G.named_parameters())):
        if n.startswith("convs"):
            layer = int(n.split(".")[1]) + 1
            params[layer][n] = p
        elif n.startswith("to_rgbs"):
            layer = int(n.split(".")[1]) * 2 + 3
            params[layer][n] = p  
        elif n.startswith("conv1"): 
            params[0][n] = p
        elif n.startswith("to_rgb1"):
            params[1][n] = p
        else:
            params["others"][n] = p
    return params

# blend the ffhq stylegan model and the finetuned model for toonify
# see ``Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains''
def blend_models(G_low, G_high, weight=[1]*7+[0]*11):
    params_low = gather_params(G_low)
    params_high = gather_params(G_high)

    for res in range(18):
        for n, p in params_high[res].items():
            params_high[res][n] = params_high[res][n] * (1-weight[res]) + params_low[res][n] * weight[res]

    state_dict = {}
    for _, p in params_high.items():
        state_dict.update(p)
        
    return state_dict

# for flask server
def decode_received_image_data(img_data):
  data = np.frombuffer(img_data, np.uint8)
  img = cv2.imdecode(data, cv2.IMREAD_COLOR)
  return img

def encode_image_to_bytes(fmt, image):
  _, encoded_image = cv2.imencode(fmt, np.asarray(image))
  return encoded_image.tobytes()
