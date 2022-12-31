"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs


class StableDiffusion(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(
        self,
        to_screen=False,
        huggingface_hub_token="huggingface_hub_token",
        unet_height=512,
        unet_width=512,
        num_inference_steps=10,
        unet_num_inference_steps=10,
        unet_latents=None,
        unet_guidance_scale=7.5,
        apply_to_latents=10,
        apply_to_embeds=10,
        clip_prompt="villa by the sea in florence on a sunny day",
        output_path="",
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value
        if True:
            self._to_screen = to_screen
            self._huggingface_hub_token = huggingface_hub_token
            self._unet_height = unet_height
            self._unet_width = unet_width
            self._num_inference_steps = num_inference_steps
            self._unet_num_inference_steps = unet_num_inference_steps
            self._unet_latents = unet_latents
            self._unet_guidance_scale = unet_guidance_scale
            self._apply_to_latents = apply_to_latents
            self._apply_to_embeds = apply_to_embeds
            self._clip_prompt = clip_prompt
            self._output_path = output_path

            from datetime import datetime

            now = datetime.now()
            self._dt_string = now.strftime("%Y.%m.%d-%H.%M.%S")

            import os.path

            if os.path.isfile(self._huggingface_hub_token):
                with open(self._huggingface_hub_token, "r") as file:
                    self._huggingface_hub_token = file.read().replace("\n", "")
            else:
                self._huggingface_hub_token = self._huggingface_hub_token

        self._to_sum_embeds = None
        self._to_sum_latents = None

        if True:

            import requests
            import torch

            torch.cuda.empty_cache()
            # import google.colab.output
            from torch import autocast
            from torch.nn import functional as F
            from torchvision import transforms
            from diffusers import (
                StableDiffusionPipeline,
                AutoencoderKL,
                UNet2DConditionModel,
                PNDMScheduler,
                LMSDiscreteScheduler,
            )
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                CLIPProcessor,
                CLIPModel,
            )
            from tqdm.auto import tqdm
            from huggingface_hub import notebook_login
            from PIL import Image, ImageDraw

            self._device = "cuda"

            # google.colab.output.enable_custom_widget_manager()
            notebook_login()

        if True:

            # Custom Pipeline
            self._vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=self._huggingface_hub_token,
            )
            self._vae = self._vae.to(self._device)  # 1GB

            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            text_encoder = text_encoder.to(self._device)  # 1.5 GB with VAE

            self._unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="unet",
                use_auth_token=self._huggingface_hub_token,
            )
            self._unet = self._unet.to(self._device)  # 4.8 GB with VAE and CLIP text

        if True:

            dict(self._vae.config)

        if True:

            def preprocess(pil_image):
                pil_image = pil_image.convert("RGB")
                processing_pipe = transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
                tensor = processing_pipe(pil_image)
                tensor = tensor.reshape(1, 3, 512, 512)
                return tensor

            self._preprocess = preprocess
        if True:

            def encode_vae(img):
                img_tensor = preprocess(img)
                with torch.no_grad():
                    diag_gaussian_distrib_obj = vae.encode(
                        img_tensor.to(device), return_dict=False
                    )
                    img_latent = diag_gaussian_distrib_obj[0].sample().detach().cpu()
                    img_latent *= 0.18215
                return img_latent

            self._encode_vae = encode_vae

        if True:

            def decode_latents(latents):
                latents = 1 / 0.18215 * latents

                with torch.no_grad():
                    images = vae.decode(latents)["sample"]

                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (images * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                return pil_images

            self._decode_latents = decode_latents

        if True:

            # CLIP
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

        if True:

            # UNet
            dict(self._unet.config)

        if True:

            # Pipeline
            self._scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

            self._scheduler1 = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

        if True:

            def get_text_embeds(prompt):
                text_input = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    text_embeddings = text_encoder(
                        text_input.input_ids.to(self._device)
                    )[0]

                uncond_input = tokenizer(
                    [""] * len(prompt),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    uncond_embeddings = text_encoder(
                        uncond_input.input_ids.to(self._device)
                    )[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                return text_embeddings

            self._get_text_embeds = get_text_embeds

            prompt = self._clip_prompt
            self._test_embeds = get_text_embeds([prompt])
            print(self._test_embeds)
            print(self._test_embeds.shape)

        if True:

            if self._unet_latents is None:
                self._unet_latents = torch.randn(
                    (
                        self._test_embeds.shape[0] // 2,
                        self._unet.in_channels,
                        self._unet_height // 8,
                        self._unet_width // 8,
                    )
                )
            #      with open('unet_latents', 'w') as file:
            #        file.write(unet_latents)
            else:
                if os.path.isfile(self._unet_latents):
                    with open(self._unet_latents, "r") as file:
                        self._unet_latents = file.read().replace("\n", "")
                else:
                    self._unet_latents = self._unet_latents

            print(self._unet_latents)
            print(self._unet_latents.shape)

            def generate_latents(
                text_embeddings,
                #        height=128,
                #        width=128,
                #        height=256,
                #        width=256,
                #        height=384,
                #        width=384,
                height=512,
                width=512,
                #        num_inference_steps=5,
                #        num_inference_steps=10,
                #        num_inference_steps=25,
                num_inference_steps=50,
                guidance_scale=7.5,
                latents=None,
            ):
                if latents is None:
                    latents = torch.randn(
                        (
                            text_embeddings.shape[0] // 2,
                            self._unet.in_channels,
                            height // 8,
                            width // 8,
                        )
                    )

                latents = latents.to(self._device)

                self._scheduler.set_timesteps(self._num_inference_steps)
                latents = latents * self._scheduler.sigmas[0]

                with autocast("cuda"):
                    for i, t in tqdm(enumerate(self._scheduler.timesteps)):
                        latent_model_input = torch.cat([latents] * 2)
                        sigma = self._scheduler.sigmas[i]
                        latent_model_input = latent_model_input / (
                            (sigma**2 + 1) ** 0.5
                        )

                        with torch.no_grad():
                            noise_pred = self._unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=text_embeddings,
                            )["sample"]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                        latents = self._scheduler.step(noise_pred, i, latents)[
                            "prev_sample"
                        ]

                return latents

            self._generate_latents = generate_latents

        if True:

            def decode_latents(latents):
                latents = 1 / 0.18215 * latents

                with torch.no_grad():
                    images = self._vae.decode(latents)["sample"]

                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (images * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                return pil_images

            self._decode_latents = decode_latents

        if True:

            def generate(
                prompts,
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=7.5,
                latents=None,
            ):
                if isinstance(prompts, str):
                    prompts = [prompts]

                text_embeds = self._get_text_embeds(prompts)
                latents = self._generate_latents(
                    text_embeds,
                    height=height,
                    width=width,
                    latents=latents,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

                images = self._decode_latents(latents)
                return images

            self._generate = generate

        latents = self._unet_latents.detach().clone()
        latents = latents.to(self._device)
        self._scheduler.set_timesteps(self._num_inference_steps)
        latents = latents * self._scheduler.sigmas[0]
        self._latentsa = [{}] * self._num_inference_steps
        # for i in range(len(latentsa)):
        #  latentsa[i]=empty
        # print(scheduler.timesteps)
        # print(tqdm(enumerate(scheduler.timesteps)))
        self._unet_latents = self._unet_latents.to(self._device)
        self._i_tqdm = 0
        self._latentsa[self._i_tqdm] = latents.detach().clone()

        if True:
            import numpy as np
            import pyformulas as pf
        if self._to_screen:
            self._canvas = np.zeros((800, 800))
            self._screen = pf.screen(self._canvas, "stylegan3")

    def update(self):
        # Make sure we have a non-empty dataframe
        if self.i.ready():

            #        self.o.data = self.i.data.tail(1)

            self.i.data

            if True:

                import requests
                import torch

                #    torch.cuda.empty_cache()
                # import google.colab.output
                from torch import autocast
                from torch.nn import functional as F
                from torchvision import transforms
                from diffusers import (
                    StableDiffusionPipeline,
                    AutoencoderKL,
                    UNet2DConditionModel,
                    PNDMScheduler,
                    LMSDiscreteScheduler,
                )
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                from transformers import (
                    CLIPTextModel,
                    CLIPTokenizer,
                    CLIPProcessor,
                    CLIPModel,
                )
                from tqdm.auto import tqdm
                from huggingface_hub import notebook_login
                from PIL import Image, ImageDraw

                import numpy as np
                import pyformulas as pf

                import pandas as pd
                import numpy as np

            if True:

                t_tqdm = self._scheduler.timesteps[self._i_tqdm]
                #  i_tqdm1=i_tqdm
                #  t_tqdm1=t_tqdm

                if True:
                    #          if False:
                    ##            cons=np.roll(cons,1,axis=0)
                    #            cons[1:,:] = cons[:len(cons),:]
                    ##            cons[0]=con[(cohs_tril_indices[0],cohs_tril_indices[1])].flatten('F')
                    #            print('_')
                    #            car_latent = encode_vae(car_img)
                    #            car_latent.shape

                    #            base_latents = test_latents.detach().clone()
                    #            base_latents = latents.detach().clone()

                    #            if True:
                    if False:
                        #              text_embeddings1=test_embeds.detach().clone()
                        #              height1=512
                        #              width1=512
                        #              num_inference_steps1=50
                        #              guidance_scale1=7.5
                        #              if i_tqdm==0:
                        #                latents1=latents.detach().clone()
                        #              if i_tqdm<num_inference_steps-1:
                        #                i_tqdm1=-1
                        #              if i_tqdm==num_inference_steps-1:
                        #              print((latentsa[i_tqdm]))
                        #              print((latentsa[i_tqdm] is None))
                        #              if not (latentsa[i_tqdm] is empty):
                        base_latents = self._latentsa[self._i_tqdm].detach().clone()
                        # base_latents.to('cpu')
                    if True:
                        base_latents = self._unet_latents.detach().clone()
                    #            base_latents = unet_and_test_draw_latents.detach().clone()

                    #            cons_latents = base_latents
                    data = self.i.data.to_numpy()
                    cons_latents_flatten = base_latents.reshape(
                        len(base_latents)
                        * len(base_latents[0])
                        * len(base_latents[0][0])
                        * len(base_latents[0][0][0])
                    )

                    cons_index = 0
                    while cons_index < len(cons_latents_flatten):
                        if self.ports is not None:
                            for name, port in self.ports.items():
                                if port.data is not None:
                                    for (colname, colval) in port.data.items():
                                        val = colval.values[len(colval.values) - 1]
                                        if cons_index < len(cons_latents_flatten):
                                            cons_latents_flatten[cons_index] = (
                                                (val + self._apply_to_latents) / 1
                                                + 0.0001
                                            ) / (
                                                (1 + self._apply_to_latents) / 1
                                                + 0.0001
                                            )
                                            cons_index = cons_index + 1
                    cons_latents = cons_latents_flatten.reshape(
                        len(base_latents),
                        len(base_latents[0]),
                        len(base_latents[0][0]),
                        len(base_latents[0][0][0]),
                    )

                if True:
                    #          if False:

                    #            base_embeds = torch.randn(2, 77, 768).to(device)
                    base_embeds = self._test_embeds.detach().clone()
                    # base_embeds = car_embeds.detach().clone()
                    cons_embeds_flatten = base_embeds.reshape(2 * 77 * 768)
                    #            print(int(len(cons_latent_flatten)/len(cons[0])))
                    cons_index = 0
                    while cons_index < len(cons_embeds_flatten):
                        if self.ports is not None:
                            for name, port in self.ports.items():
                                if port.data is not None:
                                    for (colname, colval) in port.data.items():
                                        val = colval.values[len(colval.values) - 1]
                                        if cons_index < len(cons_embeds_flatten):
                                            cons_embeds_flatten[cons_index] = (
                                                (val + self._apply_to_embeds) / 1
                                                + 0.0001
                                            ) / (
                                                (1 + self._apply_to_embeds) / 1 + 0.0001
                                            )
                                            cons_index = cons_index + 1
                    cons_embeds = cons_embeds_flatten.reshape(2, 77, 768)

                if True:
                    #          if False:

                    if self._to_sum_latents is None:
                        #            if to_sum_embeds is None:
                        #                base_latents = test_latents.detach().clone()
                        #            cons_img=Image.fromarray(cons)
                        #            cons_img_resize=cons_img.resize((400, 416))
                        #            cons_latent = encode_vae(cons_img_resize)
                        # print(car_latent)
                        #            print(cons_latent)

                        #                unet_latents
                        #                cons_latents
                        #                unet_latents.to(device)
                        #                cons_latents.to(device)

                        self._to_sum_latents = self._unet_latents / cons_latents
                        #                to_sum_latents = unet_latents.to(device)/cons_latents.to(device)
                        #                to_sum_embeds = test_embeds-cons_embeds
                        #                if False:
                        if True:
                            self._to_sum_embeds = self._test_embeds / cons_embeds
                    #                to_sum_embeds = car_embeds/cons_embeds

                    #            cons_latents.to(device)
                    #            cons_latents
                    #            sum_latents = cons_latents.to(device)*to_sum_latents.to(device)
                    sum_latents = cons_latents * self._to_sum_latents
                    #            sum_latents = latents.to(device)
                    #            sum_embeds = cons_embeds+to_sum_embeds
                    #            if False:
                    if True:
                        sum_embeds = cons_embeds * self._to_sum_embeds
                #            sum_embeds = test_embeds
                # sum_embeds = cons_embeds
                ##            sum_embeds = test_embeds
                #            sum_latent = car_latent
                ##            test_latents = generate_latents(sum_embeds)
                #            test_latents = generate_latents(test_embeds)

                #            latents = sum_latents

                if True:

                    #            if True:
                    if False:

                        #              text_embeddings1=sum_embeds.detach().clone()
                        text_embeddings = self._test_embeds
                        #              text_embeddings1=test_embeds.detach().clone()
                        #              latents1=sum_latents
                        #              latents1=latentsa[i_tqdm].detach().clone()
                        latents = self._latentsa[self._i_tqdm].detach().clone()
                        #              height1=512
                        #              width1=512
                        #              num_inference_steps1=50
                        guidance_scale1 = self._unet_guidance_scale
                        #              if i_tqdm==0:
                        #                latents1=latents.detach().clone()
                        #              if i_tqdm<num_inference_steps-1:
                        #                i_tqdm1=-1
                        #              if i_tqdm==num_inference_steps-1:
                        #              print((latentsa[i_tqdm]))
                        #              print((latentsa[i_tqdm] is None))
                        #              if not (latentsa[i_tqdm] is empty):
                        #              latents1=latentsa[i_tqdm].detach().clone()
                        #              latents1=unet_latents.detach().clone()
                        #              latents1=None
                        #              if latents1 is None:
                        #                  latents1 = torch.randn((
                        #                      text_embeddings1.shape[0] // 2,
                        #                      unet.in_channels,
                        #                      height1 // 8,
                        #                      width1 // 8
                        #                  ))

                        #              latents1 = latents1.to(device)

                        #              scheduler1.set_timesteps(num_inference_steps1)
                        #              latents1 = latents1 * scheduler1.sigmas[0]

                        if True:
                            with autocast("cuda"):
                                #                  for i1, t1 in tqdm(enumerate(scheduler1.timesteps)):
                                i1 = self._i_tqdm
                                t1 = self._t_tqdm
                                #                      scheduler1=scheduler
                                latent_model_input = torch.cat([latents] * 2)
                                sigma = self._scheduler.sigmas[i1]
                                latent_model_input = latent_model_input / (
                                    (sigma**2 + 1) ** 0.5
                                )
                                #                      print(i1,t1,sigma1)

                                with torch.no_grad():
                                    noise_pred = self._unet(
                                        latent_model_input,
                                        t1,
                                        encoder_hidden_states=text_embeddings,
                                    )["sample"]

                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale1 * (
                                    noise_pred_text - noise_pred_uncond
                                )

                                latents = self._scheduler.step(noise_pred, i1, latents)[
                                    "prev_sample"
                                ]

                                #                      latents=latents1.detach().clone()
                                #                      scheduler=scheduler1

                                if True:

                                    self._i_tqdm = self._i_tqdm + 1
                                    if self._i_tqdm < num_inference_steps:
                                        #                          if (latentsa[i_tqdm] is empty):
                                        #                            latentsa[i_tqdm]=latents.detach().clone()
                                        self._latentsa[
                                            self._i_tqdm
                                        ] = latents.detach().clone()
                                    if self._i_tqdm == self._num_inference_steps:
                                        #                          i_tqdm=random.randint(0,num_inference_steps-1)
                                        self._i_tqdm = 1
                                    #                          i_tqdm=0
                                    #                          for i in range(i_tqdm+1,len(latentsa)):
                                    #                            latentsa[i]=empty

                                    self._test_latents = latents
                                #                        test_latents = latents.detach().clone()
                                #                       test_latents = latents1.detach().clone()

                                if False:

                                    images = decode_latents(test_latents.to(device))
                                    images[0].save(
                                        output_path
                                        + "stable-diffusion-"
                                        + dt_string
                                        + "-"
                                        + FLAGS.clip_prompt
                                        + ".png",
                                        format="png",
                                    )
                                    images[0].save("mygraph.png", format="png")
                                    image = np.asarray(images[0])
                                    image = image[:, :, ::-1]
                                    screen3.update(image)

                    #              test_latents = latents1.detach().clone()
                    #              test_latents = latents.detach().clone()

                    #            if True:
                    if False:

                        latent_model_input = torch.cat([latents] * 2)
                        sigma = self._scheduler.sigmas[self._i_tqdm]
                        latent_model_input = latent_model_input / (
                            (sigma**2 + 1) ** 0.5
                        )

                        with torch.no_grad():
                            noise_pred = unet(
                                latent_model_input,
                                t_tqdm,
                                encoder_hidden_states=test_embeds.clone(),
                            )["sample"]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + unet_guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                        latents = self._scheduler.step(noise_pred, i_tqdm, latents)[
                            "prev_sample"
                        ]
                        test_latents = latents.clone()

                    #            sum_embeds = test_embeds
                    #            sum_latents = unet_latents

                    if True:
                        #            if False:

                        test_latents = self._generate_latents(
                            sum_embeds,
                            #                  test_embeds,
                            height=self._unet_height,
                            width=self._unet_width,
                            num_inference_steps=self._unet_num_inference_steps,
                            #                  latents=unet_latents,
                            #                  latents=cons_latents,
                            latents=sum_latents,
                            guidance_scale=self._unet_guidance_scale,
                        )

                    images = self._decode_latents(test_latents.to(self._device))
                    # images = decode_latents(car_latent.to(device))
                    # print(images[0])
                    images[0].save(
                        self._output_path
                        + "stable-diffusion-"
                        + self._dt_string
                        + "-"
                        + self._clip_prompt
                        + ".png",
                        format="png",
                    )
                    images[0].save("mygraph.png", format="png")
                    #            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                    #            fig = plt.figure(figsize=(800*px, 800*px))
                    #            plt.imshow(images[0])
                    #            plt.close()
                    #            fig.canvas.draw()
                    #            image = np.frombuffer(fig.canvas.tostring_rgb(),'u1')
                    #            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    image = np.asarray(images[0])
                    image = image[:, :, ::-1]

            if self._to_screen:
                self._screen.update(image)


#            video_outs[shows_idx].append_data(image)
