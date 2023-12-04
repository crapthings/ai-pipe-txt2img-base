import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download('Bingsu/adetailer', 'face_yolov8n.pt')
detector = YOLO(path)

def fix_face (input_image):
    result = detector(input_image)[0]

    if result.masks is None:
        return input_image

    bboxes = result.boxes.xyxy.cpu().numpy()

    face_mask = create_mask_from_bbox(bboxes, input_image.size)[0]

    face_mask = face_mask.convert('L')
    face_mask = mask_dilate(face_mask, 4)
    face_mask = mask_gaussian_blur(face_mask, 4)

    bbox_padded = bbox_padding(face_mask.getbbox(), input_image.size, 32)

    crop_image = input_image.crop(bbox_padded)
    crop_mask = face_mask.crop(bbox_padded)

    width, height = round_to_nearest_multiple_of_eight(crop_image.width * 4, crop_image.height * 4)

    # after detailer
    face = pipe(
        prompt,
        negative_prompt = negative_prompt,
        image = crop_image,
        mask_image = crop_mask,
        width = width,
        height = height,
        strength = 1,
        num_inference_steps = 20,
        guidance_scale = .5,
    ).images[0]

    face = face.resize(crop_image.size)

    input_image = composite(input_image, face_mask, face, bbox_padded)

    return input_image

def bbox_padding (bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32) -> tuple[int, int, int, int]:
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.flatten())

def round_to_nearest_multiple_of_eight (width, height):
    rounded_width = (width // 8) * 8
    rounded_height = (height // 8) * 8

    if width % 8 >= 4:
        rounded_width += 8
    if height % 8 >= 4:
        rounded_height += 8

    return int(rounded_width), int(rounded_height)

def mask_dilate (image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations = 1)
    return Image.fromarray(dilated)


def mask_gaussian_blur (image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)

def create_mask_from_bbox (bboxes: np.ndarray, shape: tuple[int, int]) -> list[Image.Image]:
    masks = []
    for bbox in bboxes:
        mask = Image.new('L', shape, 'black')
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill = 'white')
        masks.append(mask)
    return masks

def composite(init: Image.Image, mask: Image.Image, gen: Image.Image, bbox_padded: tuple[int, int, int, int]) -> Image.Image:
    img_masked = Image.new('RGBa', init.size)
    img_masked.paste(init.convert('RGBA').convert('RGBa'), mask = ImageOps.invert(mask))
    img_masked = img_masked.convert('RGBA')
    size = (bbox_padded[2] - bbox_padded[0], bbox_padded[3] - bbox_padded[1])
    resized = gen.resize(size)
    output = Image.new('RGBA', init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert('RGB')
