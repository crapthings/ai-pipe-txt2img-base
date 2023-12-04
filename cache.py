import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from compel import Compel

from config import model_name

print('cache model')

pipe = StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

pipe2 = AutoPipelineForImage2Image.from_pipe(pipe)

compel_proc = Compel(tokenizer = pipe.tokenizer, text_encoder = pipe.text_encoder)

print('done')
