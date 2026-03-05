import pytest
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image

MODEL_NAME = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"

@pytest.fixture(scope="module")
def model():
    model = StableDiffusionPipeline.from_pretrained(MODEL_NAME, device_map = None, low_cpu_mem_usage=True)
    return model

def test_model_and_processor_loading(model):
    model = model
    assert model is not None
    assert isinstance(model, StableDiffusionPipeline)
    assert model.text_encoder is not None
    assert model.unet is not None
    assert model.tokenizer is not None

def test_prompt_processing(model):
    prompt = "magical human, pixel art, white background"
    inputs = model.tokenizer(
        prompt,
        return_tensors = "pt"
    )
    assert "input_ids" in inputs
    assert inputs.input_ids.shape[1] > 0

def test_image_generation_format(model):
    prompt = "magical human, pixel art, white background"
    output = model(prompt).images[0]
    assert isinstance(output, Image.Image)
    assert output.size == (512, 512)


def test_image_is_not_empty(model):
    prompt = "magical human, pixel art, white background"
    output = model(prompt).images[0]
    img_array = np.array(output)
    assert img_array.std() > 0