import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from compel import Compel

from config import model_name

print('cache model')

text_to_image = StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16
)

image_to_image = AutoPipelineForImage2Image.from_pipe(text_to_image)

compel_proc = Compel(tokenizer = text_to_image.tokenizer, text_encoder = text_to_image.text_encoder)

print('done')
