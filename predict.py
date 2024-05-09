import os
import mimetypes
import random
import json
import shutil
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

mimetypes.add_type("image/webp", ".webp")


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open("pulid_api.json", "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(workflow)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        image = Image.open(input_file)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (KeyError, AttributeError):
            # EXIF data does not have orientation
            # Do not rotate
            pass

        image.save(os.path.join(INPUT_DIR, filename))

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def set_weights(self, workflow, model: str):
        loader = workflow["4"]["inputs"]
        model_map = {
            "general - albedobaseXL_v21": "albedobaseXL_v21.safetensors",
            "general - dreamshaperXL_alpha2Xl10": "dreamshaperXL_alpha2Xl10.safetensors",
            "animated - starlightXLAnimated_v3": "starlightXLAnimated_v3.safetensors",
            "animated - pixlAnimeCartoonComic_v10": "pixlAnimeCartoonComic_v10.safetensors",
            "realistic - rundiffusionXL_beta": "rundiffusionXL_beta.safetensors",
            "realistic - RealVisXL_V4.0": "RealVisXL_V4.0.safetensors",
            "realistic - sdxlUnstableDiffusers_nihilmania": "sdxlUnstableDiffusers_nihilmania.safetensors",
            "cinematic - CinematicRedmond": "CinematicRedmond.safetensors",
        }

        if model in model_map:
            loader["ckpt_name"] = model_map[model]

    def update_workflow(self, workflow, **kwargs):
        self.set_weights(workflow, kwargs["model"])

        workflow["22"]["inputs"]["text"] = kwargs["prompt"]
        workflow["23"]["inputs"]["text"] = f"nsfw, nude, {kwargs['negative_prompt']}"

        if kwargs["face_style"] == "high-fidelity":
            workflow["33"]["inputs"]["method"] = "fidelity"
        else:
            workflow["33"]["inputs"]["method"] = "style"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

    def predict(
        self,
        face_image: Path = Input(
            description="The face image to use for the generation",
        ),
        prompt: str = Input(
            description="You might need to include a gender in the prompt to get the desired result",
            default="A photo of a person",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        width: int = Input(
            description="Width of the output image (ignored if structure image given)",
            default=1024,
        ),
        height: int = Input(
            description="Height of the output image (ignored if structure image given)",
            default=1024,
        ),
        checkpoint_model: str = Input(
            description="Model to use for the generation",
            choices=[
                "general - albedobaseXL_v21",
                "general - dreamshaperXL_alpha2Xl10",
                "animated - starlightXLAnimated_v3",
                "animated - pixlAnimeCartoonComic_v10",
                "realistic - rundiffusionXL_beta",
                "realistic - RealVisXL_V4.0",
                "realistic - sdxlUnstableDiffusers_nihilmania",
                "cinematic - CinematicRedmond",
            ],
            default="general - dreamshaperXL_alpha2Xl10",
        ),
        face_style: str = Input(
            description="Style of the face",
            choices=["high-fidelity", "stylized"],
            default="high-fidelity",
        ),
        number_of_images: int = Input(
            description="Number of images to generate", default=1, ge=1, le=10
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        if not face_image:
            raise ValueError("Style image is required")

        self.handle_input_file(face_image)

        with open("pulid_api.json", "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            model=checkpoint_model,
            face_style=face_style,
            number_of_images=number_of_images,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        files = self.log_and_collect_files(OUTPUT_DIR)

        if output_quality < 100 or output_format in ["webp", "jpg"]:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(f".{output_format}")
                    image.save(
                        optimised_file_path,
                        quality=output_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
