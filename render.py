import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler
from compel import Compel

from config import model_name
from utils import sc

safety_checker.StableDiffusionSafetyChecker.forward = sc

# text to image

text_to_image_pipeline = StableDiffusionPipeline.from_single_file(
  model_name,
  torch_dtype = torch.float16
)

text_to_image_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(text_to_image_pipeline.scheduler.config)

text_to_image_pipeline.enable_freeu(s1 = 0.9, s2 = 0.2, b1 = 1.2, b2 = 1.4)

# text_to_image_pipeline.to('cuda')
text_to_image_pipeline.enable_model_cpu_offload()

# image to image pipeline

image_to_image_pipeline = AutoPipelineForImage2Image.from_pipe(text_to_image_pipeline)

image_to_image_pipeline.enable_freeu(s1 = 0.9, s2 = 0.2, b1 = 1.2, b2 = 1.4)

# prompt weighting

prompt_encoder = Compel(tokenizer = text_to_image_pipeline.tokenizer, text_encoder = text_to_image_pipeline.text_encoder)

# render

def text_to_image (**props):
  output = text_to_image_pipeline(**props)
  return output

def image_to_image (**props):
  output = image_to_image_pipeline(**props)
  return output

# sampler

sampler_dict = {
  'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
  'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
  'HeunDiscreteScheduler': HeunDiscreteScheduler,
  'LMSDiscreteScheduler': LMSDiscreteScheduler,
  'UniPCMultistepScheduler': UniPCMultistepScheduler
}

def set_sampler (scheduler_name):
  sampler = sampler_dict.get(scheduler_name, EulerAncestralDiscreteScheduler)
  text_to_image_pipeline.scheduler = sampler.from_config(text_to_image_pipeline.scheduler.config)
  image_to_image_pipeline.scheduler = sampler.from_config(image_to_image_pipeline.scheduler.config)

  if scheduler_name == 'DPMSolverMultistepScheduler':
    text_to_image_pipeline.scheduler = sampler.from_config(text_to_image_pipeline.scheduler.config, algorithm_type = 'sde-dpmsolver++', use_karras_sigmas = True)
    image_to_image_pipeline.scheduler = sampler.from_config(image_to_image_pipeline.scheduler.config, algorithm_type = 'sde-dpmsolver++', use_karras_sigmas = True)

  elif scheduler_name == 'LMSDiscreteScheduler':
    text_to_image_pipeline.scheduler = sampler.from_config(text_to_image_pipeline.scheduler.config, use_karras_sigmas = True)
    image_to_image_pipeline.scheduler = sampler.from_config(image_to_image_pipeline.scheduler.config, use_karras_sigmas = True)
