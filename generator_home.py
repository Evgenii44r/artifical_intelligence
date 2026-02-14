from diffusers import DiffusionPipeline
from PIL import Image
import torch
import time


def generate_fast():
    print("Загрузка легкой модели...")

    # Используем маленькую модель (работает быстрее на CPU)
    pipe = DiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        safety_checker=None,
        requires_safety_checker=False
    )

    # Оптимизация
    pipe.enable_attention_slicing()
    torch.set_num_threads(4)
    pipe = pipe.to("cpu")

    print("Генерация...")
    image = pipe(
        "a cat sitting on a bench",
        num_inference_steps=15,
        height=256,
        width=256
    ).images[0]

    # Увеличиваем
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image.save("fast_cat.png")
    print("Готово! Изображение сохранено как fast_cat.png")


if __name__ == "__main__":
    generate_fast()