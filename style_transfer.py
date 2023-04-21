import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_psp_standalone, get_video_crop_parameter, tensor2cv2, get_crop_parameter_by_mediapipe, creat_weight_kernel, create_weight_field
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from model.encoder.encoders.psp_encoders import GradualStyleEncoder
from matting.rembg_simplify import remove
from mediapipe.python.solutions.face_detection import FaceDetection
import time

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image/video")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--video", action="store_true", help="if true, video stylization; if false, image stylization")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--parsing_map_path", type=str, default=None, help="path of the refined parsing map of the target video")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def create_image_style_transfer_dualstylegan_models(
        style_id: int = 299,
        device: str = 'cuda',
        ckpt: str = './checkpoint/vtoonify_d_cartoon/vtoonify_s299_d0.5.pt',
        faceparsing_ckpt: str = './checkpoint/faceparsing.pth',
        pspencoder_ckpt: str = './checkpoint/encoder.pt',
        exstyle_path: str = './checkpoint/vtoonify_d_cartoon/exstyle_code.npy',    # usually in the same dir with ckpt
        ):
    vtoonify = VToonify(backbone = 'dualstylegan')
    print('loading ckpt: {}'.format(ckpt))
    vtoonify.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(faceparsing_ckpt, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    pspencoder = load_psp_standalone(pspencoder_ckpt, device)

    exstyles = np.load(exstyle_path, allow_pickle='TRUE').item()
    stylename = list(exstyles.keys())[style_id]
    exstyle = torch.tensor(exstyles[stylename]).to(device)
    with torch.no_grad():  
        exstyle = vtoonify.zplus2wplus(exstyle)

    return vtoonify, parsingpredictor, pspencoder, exstyle

def image_style_transfer_dualstylegan(
    frame: np.ndarray,
    style_id: int = 299,
    device: str = 'cuda',
    padding: List[int] = [120, 120, 120, 120],
    faceDetector: Optional[FaceDetection] = None,
    models: Optional[Tuple[VToonify, BiSeNet, GradualStyleEncoder, torch.Tensor]] = None,
) -> Optional[np.ndarray]:
    if models is None:
        vtoonify, parsingpredictor, pspencoder, exstyle = create_image_style_transfer_dualstylegan_models(style_id, device)
    else:
        vtoonify, parsingpredictor, pspencoder, exstyle = models

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])

    # resize, longest edge of frame is not greater than 1k
    H, W = frame.shape[:2]
    if max(H, W) > 1024:
        ratio = 1024 / max(H, W)
        frame = cv2.resize(frame, (round(W * ratio), round(H * ratio)))
    origin = frame.copy()

    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    if faceDetector is None:
        faceDetector = FaceDetection(min_detection_confidence=0.5)
    crop_paras = get_crop_parameter_by_mediapipe(frame, faceDetector, padding)

    if crop_paras is None:
        return None
    else:
        h,w,top,bottom,left,right = crop_paras
        frame = cv2.resize(frame[top:bottom, left:right], (w, h))

        # we apply gaussian blur to it to avoid over-sharp stylization results
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        with torch.no_grad():
            I = transform(frame).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)
            if vtoonify.backbone == 'dualstylegan':
                s_w[:,:7] = exstyle[:,:7]

            x = transform(frame).unsqueeze(dim=0).to(device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0],
                                scale_factor=0.5, recompute_scale_factor=False).detach()
            torch.cuda.empty_cache()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p/16.), dim=1)
            # d_s has no effect when backbone is toonify
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = 0.5)
            y_tilde = torch.clamp(y_tilde, -1, 1)

        output = (y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) * 0.5
        blend = blending(origin, output, top, bottom, left, right)
        return blend

def blending(origin: np.ndarray, output: np.ndarray, top: int, bottom: int, left: int, right: int):
    output = cv2.resize(output, (right - left, bottom - top))
    if origin.max() <= 1:
        origin = (origin * 255).astype(np.uint8)
    if output.max() <= 1:
        output = (output * 255).astype(np.uint8)

    # matte the human part and do blending
    mask_output = remove(output, only_mask=True)
    mask_origin = remove(origin, only_mask=True)
    mask = np.where((mask_output + mask_origin[top:bottom, left:right]) > 10, 255 * np.ones_like(mask_output), np.zeros_like(mask_output))

    # the hair part changes a lot and face part may shrink, so I dilate the mask
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations = 10)

    # do the boundRect to get corrent center, because:
    # seamlessClone will do boundRect crop and place the cropped source image on the destination image,
    # where the Point p is the center of the cropped source (after margin removal) - and not the center of the original source.
    # see https://github.com/opencv/opencv/issues/21902
    x, y, rect_w, rect_h = cv2.boundingRect(mask)
    center = (int(left + x + rect_w / 2), int(top + y + rect_h / 2))

    # we apply gaussian blur to origin by its max edge
    # I have to do gaussian blur because background of output region is much more blur than origin
    origin_max_edge = max(*origin.shape[:2])
    if origin_max_edge > 768:
        origin_blur = cv2.GaussianBlur(origin, (5, 5), 0)
    elif origin_max_edge > 512:
        origin_blur = cv2.GaussianBlur(origin, (3, 3), 0)
    else:
        origin_blur = origin.copy()

    blend = cv2.seamlessClone(output, origin_blur, mask, center, cv2.NORMAL_CLONE)

    # smooth the edge region of mask
    weight = np.zeros_like(blend)[..., 0]
    weight[top:bottom, left:right] = mask
    weight_field = create_weight_field(weight, kernel_size=(5, 5), iterations=30, a=1.1)
    if len(weight_field.shape) == 2:
        weight_field = weight_field[..., np.newaxis]
    blend = (blend / 255.) * weight_field + (origin_blur / 255.) * (1 - weight_field)
    return (blend * 255).astype(np.uint8)

    
if __name__ == "__main__":

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    
    device = "cpu" if args.cpu else "cuda"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
    vtoonify = VToonify(backbone = args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    # modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    # if not os.path.exists(modelname):
    #     import wget, bz2
    #     wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
    #     zipfile = bz2.BZ2File(modelname+'.bz2')
    #     data = zipfile.read()
    #     open(modelname, 'wb').write(data) 
    # landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args.style_encoder_path, device)    

    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():  
            exstyle = vtoonify.zplus2wplus(exstyle)

    if args.video and args.parsing_map_path is not None:
        x_p_hat = torch.tensor(np.load(args.parsing_map_path))          
            
    print('Load models successfully!')
    
    
    filename = args.content
    basename = os.path.basename(filename).split('.')[0]
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
    print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
    if args.video:
        cropname = os.path.join(args.output_path, basename + '_input.mp4')
        savename = os.path.join(args.output_path, basename + '_vtoonify_' +  args.backbone[0] + '.mp4')

        video_cap = cv2.VideoCapture(filename)
        num = int(video_cap.get(7))

        first_valid_frame = True
        batch_frames = []
        for i in tqdm(range(num)):
            success, frame = video_cap.read()
            if success == False:
                assert('load video frames error')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # We proprocess the video by detecting the face in the first frame, 
            # and resizing the frame so that the eye distance is 64 pixels.
            # Centered on the eyes, we crop the first frame to almost 400x400 (based on args.padding).
            # All other frames use the same resizing and cropping parameters as the first frame.
            if first_valid_frame:
                if args.scale_image:
                    paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
                    if paras is None:
                        continue
                    h,w,top,bottom,left,right,scale = paras
                    H, W = int(bottom-top), int(right-left)
                    # for HR video, we apply gaussian blur to the frames to avoid flickers caused by bilinear downsampling
                    # this can also prevent over-sharp stylization results. 
                    if scale <= 0.75:
                        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                    if scale <= 0.375:
                        frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                    frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
                else:
                    H, W = frame.shape[0], frame.shape[1]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                videoWriter = cv2.VideoWriter(cropname, fourcc, video_cap.get(5), (W, H))
                videoWriter2 = cv2.VideoWriter(savename, fourcc, video_cap.get(5), (4*W, 4*H))
                
                # For each video, we detect and align the face in the first frame for pSp to obtain the style code. 
                # This style code is used for all other frames.
                with torch.no_grad():
                    I = align_face(frame, landmarkpredictor)
                    I = transform(I).unsqueeze(dim=0).to(device)
                    s_w = pspencoder(I)
                    s_w = vtoonify.zplus2wplus(s_w)
                    if vtoonify.backbone == 'dualstylegan':
                        if args.color_transfer:
                            s_w = exstyle
                        else:
                            s_w[:,:7] = exstyle[:,:7]
                first_valid_frame = False
            elif args.scale_image:
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

            videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            batch_frames += [transform(frame).unsqueeze(dim=0).to(device)]

            if len(batch_frames) == args.batch_size or (i+1) == num:
                x = torch.cat(batch_frames, dim=0)
                batch_frames = []
                with torch.no_grad():
                    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
                    # followed by downsampling the parsing maps
                    if args.video and args.parsing_map_path is not None:
                        x_p = x_p_hat[i+1-x.size(0):i+1].to(device)
                    else:
                        x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                        scale_factor=0.5, recompute_scale_factor=False).detach()
                    # we give parsing maps lower weight (1/16)
                    inputs = torch.cat((x, x_p/16.), dim=1)
                    # d_s has no effect when backbone is toonify
                    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = args.style_degree)       
                    y_tilde = torch.clamp(y_tilde, -1, 1)
                for k in range(y_tilde.size(0)):
                    videoWriter2.write(tensor2cv2(y_tilde[k].cpu()))

        videoWriter.release()
        videoWriter2.release()
        video_cap.release()

    
    else:
        cropname = os.path.join(args.output_path, basename + '_input.jpg')
        savename = os.path.join(args.output_path, basename + '_vtoonify_' +  args.backbone[0] + '.jpg')

        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        origin = frame.copy()

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if args.scale_image:
            # paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
            paras = get_crop_parameter_by_mediapipe(frame, args.padding)
            if paras is not None:
                h,w,top,bottom,left,right,scale = paras
                H, W = int(bottom-top), int(right-left)
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
        
        print('start inference')

        with torch.no_grad():
            start = time.time()
            # h, w, _ = frame.shape
            # frame = cv2.resize(frame, (w // 8 * 8, h // 8 * 8))
            
            # I = align_face(frame, landmarkpredictor)
            # I = transform(I).unsqueeze(dim=0).to(device)
            
            I = transform(frame).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)
            if vtoonify.backbone == 'dualstylegan':
                if args.color_transfer:
                    s_w = exstyle
                else:
                    s_w[:,:7] = exstyle[:,:7]

            x = transform(frame).unsqueeze(dim=0).to(device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                scale_factor=0.5, recompute_scale_factor=False).detach()
            torch.cuda.empty_cache()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p/16.), dim=1)
            # d_s has no effect when backbone is toonify
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = args.style_degree)        
            y_tilde = torch.clamp(y_tilde, -1, 1)

            end = time.time()
            print('time = ', end - start)

        cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        save_image(y_tilde[0].cpu(), savename)

        if args.scale_image:
            if paras is not None:
                H, W, _ = origin.shape
                h,w,top,bottom,left,right,scale = paras
                origin_copy = cv2.resize(origin  / 255., (w, h))
                output = (y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) * 0.5
                output = cv2.resize(output, (right - left, bottom - top))
                weight_kernel = (creat_weight_kernel((right - left, bottom - top)))[..., np.newaxis]

                origin_copy[top:bottom, left:right] = output * weight_kernel + origin_copy[top:bottom, left:right] * (1 - weight_kernel)
                origin_copy = cv2.resize(origin_copy, (W, H))
                origin_copy = (origin_copy * 255).astype(np.uint8)
                plt.imshow(origin_copy)
                plt.show()

    print('function test')
    res = image_style_transfer_cartoon299(origin.copy(), device = device, padding = args.padding) #, vtoonify, parsingpredictor, pspencoder, exstyle, device, args.padding)
    print(res.shape)

    print('Transfer style successfully!')