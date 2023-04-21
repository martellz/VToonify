from pathlib import Path
import json, base64, hashlib
from flask import Blueprint, request
from flask_cors import cross_origin
# from backend.matting.rembg_simplify import get_background_mask
# from backend.generativemodels.inpaint import create_inpaint_pipeline
# from backend.generativemodels.inpaint import inpaint
from style_transfer import create_image_style_transfer_dualstylegan_models, image_style_transfer_dualstylegan
from util import encode_image_to_bytes, decode_received_image_data
from server_config import config
from mediapipe.python.solutions.face_detection import FaceDetection

bp = Blueprint('changeBg', __name__, url_prefix='/changeBg')

ckpt_dir = config['ckpt_dir']
style_id = config['style_id']
device = config['device']
padding = config['padding']

models = create_image_style_transfer_dualstylegan_models(style_id, device)
faceDetector = FaceDetection()

@bp.route('', methods=('POST', ))
@cross_origin()
def submit_query():
  image_data = request.files['image'].read()
  image = decode_received_image_data(image_data)[:, :, [2, 1, 0]]  # BGR2RGB
  new_img = image_style_transfer_dualstylegan(image, style_id, device, [padding for _ in range(4)], faceDetector, models)[:, :, [2, 1, 0]] # RGB2BGR
  encoded_image = encode_image_to_bytes('.jpg', new_img)
  return json.dumps({
    'format': 'img/jpeg',
    'image': base64.b64encode(encoded_image).decode('utf-8')
  })
