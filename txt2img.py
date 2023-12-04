import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from diffusers import EulerAncestralDiscreteScheduler
from compel import Compel

from config import model_name

def sc(self, clip_input, images): return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

txt2imgPipe = StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

txt2imgPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(txt2imgPipe.scheduler.config)

txt2imgPipe.enable_freeu(s1 = 0.9, s2 = 0.2, b1 = 1.2, b2 = 1.4)

txt2imgPipe.to('cuda')

img2imgPipe = AutoPipelineForImage2Image.from_pipe(txt2imgPipe)

img2imgPipe.enable_freeu(s1 = 0.9, s2 = 0.2, b1 = 1.2, b2 = 1.4)

compel_proc = Compel(tokenizer = txt2imgPipe.tokenizer, text_encoder = txt2imgPipe.text_encoder)

def txt2img (**props):
  output = txt2imgPipe(**props)
  return output

def img2img (**props):
  output = img2imgPipe(**props)
  return output
