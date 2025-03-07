# -*- coding: utf-8 -*-
"""model_tryout.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QsCmg4u5ZJs_W8_eL9dZ9Nz_MXINU4rt
"""

!pip install transformers[vision]

!pip install torch transformers accelerate
!pip install git+https://github.com/huggingface/transformers

from transformers import LlavaForConditionalGeneration, LlavaProcessor

model_name = "LanguageBind/Video-LLaVA-7B"

model = LlavaForConditionalGeneration.from_pretrained(model_name)
processor = LlavaProcessor.from_pretrained(model_name)

!pip install decord

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from decord import VideoReader, cpu
import numpy as np

# Load model and processor
model_name = "microsoft/git-large-vatex"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from decord import VideoReader, cpu
import numpy as np

def video_clip_caption(video_path, num_frames=16):
    # Use correct model for caption generation
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
    processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")

    # Extract frames properly
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int32)
    frames = vr.get_batch(indices).asnumpy()

    # Process inputs correctly
    inputs = processor(images=list(frames), return_tensors="pt")

    # Generate caption
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        max_length=100,
        num_beams=4
    )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Usage
caption = video_clip_caption("/content/testvid.mp4")
print("Video Caption:", caption)

!pip install transformers[vision]
!pip install torch transformers accelerate
!pip install git+https://github.com/huggingface/transformers

from transformers import LlavaForConditionalGeneration, LlavaProcessor

def videollama_caption(video_path):
    model_name = "LanguageBind/Video-LLaVA-7B" # update with actual model name

    model = LlavaForConditionalGeneration.from_pretrained(model_name)
    processor = LlavaProcessor.from_pretrained(model_name)

    # update the rest with model's specific inference logic
    # ...

!pip install llava-python decord torch
!git clone https://github.com/haotian-liu/LLaVA.git
!cd LLaVA && pip install -e .

!pip install stability_vlm

from videoclip import VideoCLIP

model = VideoCLIP.from_pretrained("mit/videoclip-base")
def videoclip_caption(video_path):
    return model.caption_video(
        video_path,
        prompt="A video of",
        beam_size=3,
        max_length=30
    )

# Usage
caption = videoclip_caption("/content/testvid.mp4")
print(f"Video Caption: {caption}")

!pip install av

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# load video
video_path = "/content/testvid.mp4"
container = av.open(video_path)

# extract evenly spaced frames from video
seg_len = container.streams.video[0].frames
clip_len = model.config.encoder.num_frames
indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    if i in indices:
        frames.append(frame.to_ndarray(format="rgb24"))

# generate caption
gen_kwargs = {
    "min_length": 10,
    "max_length": 20,
    "num_beams": 8,
}
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
tokens = model.generate(pixel_values, **gen_kwargs)
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(caption) # A man and a woman are dancing on a stage in front of a mirror.

!pip install pllava

import torch
from pllava import PLLaVA, PLLaVAConfig
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

def pllava_video_caption(video_path, num_frames=8):
    # Initialize PLLaVA with temporal pooling
    config = PLLaVAConfig(
        model_name="pllava-7b-v1.5",
        temporal_pooling="attention",  # Options: "mean", "max", "attention"
        num_frames=num_frames
    )
    model = PLLaVA(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Extract and preprocess frames
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_indices = np.linspace(0, len(vr)-1, num_frames, dtype=int)
    frames = [Image.fromarray(fr) for fr in vr.get_batch(frame_indices).asnumpy()]

    # Generate video caption
    prompt = (
        "You are a video analysis expert. Describe this video in detail, "
        "including temporal relationships between objects and actions:"
    )

    outputs = model.generate(
        images=frames,
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        use_temporal_pooling=True
    )

    return outputs[0]

# Usage
caption = pllava_video_caption("/content/test_video.mp4")
print(f"Video Caption: {caption}")

