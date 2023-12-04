import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from compel import Compel

from config import model_name

print('cache model')

text2image = StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16
)

image2image = AutoPipelineForImage2Image.from_pipe(text2image)

compel_proc = Compel(tokenizer = text2image.tokenizer, text_encoder = text2image.text_encoder)

print('done')
