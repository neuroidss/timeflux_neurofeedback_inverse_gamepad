"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs


class StyleGAN3(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self, files_path, to_screen=False):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value
        self._to_screen = to_screen
        files_path = [files_path]

        sg3_models = 0

        #    os.chdir('/content')
        #    if generate&gen_sg3_nvlabs_pt:
        import os.path

        if not os.path.isdir("stylegan3-nvlabs-pytorch"):
            os.system(
                "git clone https://github.com/NVlabs/stylegan3.git stylegan3-nvlabs-pytorch"
            )
        #    os.chdir('/content/stylegan3-nvlabs-pytorch')
        #    if generate&gen_sg3_Expl0dingCat_pt:
        #    os.system('git clone https://github.com/Expl0dingCat/stylegan3-modified.git /content/stylegan3-Expl0dingCat-pytorch')
        #    os.chdir('/content/stylegan3-Expl0dingCat-pytorch')

        def download_file_from_google_drive(file_id, dest_path):
            import os.path

            if not os.path.isfile(dest_path):
                os.system("mkdir -p " + os.path.dirname(dest_path))
                os.system(
                    "wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='"
                    + file_id
                    + " -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt"
                )
                os.system(
                    "wget --load-cookies cookies.txt -O "
                    + dest_path
                    + " 'https://docs.google.com/uc?export=download&id='"
                    + file_id
                    + "'&confirm='$(<confirm.txt)"
                )

        #    os.system('pip install scipy')
        #    !pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
        #  !pip install torch
        #  !pip install torch==1.7.1
        #  %pip install ninja
        #  import pickle
        import copy
        import os

        # from time import perf_counter

        # import click
        import imageio
        import numpy as np
        import PIL.Image
        import torch
        import torch.nn.functional as F

        import sys

        sys.path.insert(0, "stylegan3-nvlabs-pytorch")
        #    sys.path.insert(0, '/content/stylegan3-Expl0dingCat-pytorch')

        import dnnlib
        import legacy

        if True:
            #    if False:

            for i in range(len(files_path)):
                download_file_from_google_drive(
                    file_id=files_path[i][0], dest_path=files_path[i][1]
                )

        self._G3ms = [{}] * len(files_path)
        for i in range(len(files_path)):
            network_pkl = files_path[i][1]
            self._device = torch.device("cuda")
            with dnnlib.util.open_url(network_pkl) as fp:
                #      G3m = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
                self._G3ms[i] = legacy.load_network_pkl(fp)["G_ema"].to(self._device)  # type: ignore

        #    if draw_fps:
        #      time001a=[{}]*len(files_path)
        #      time111a=[{}]*len(files_path)
        #      for i in range(len(files_path)):
        #        time001a[i]=perf_counter()

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

            import imageio
            import numpy as np
            import PIL.Image
            import torch
            import torch.nn.functional as F

            import sys

            sys.path.insert(0, "stylegan3-nvlabs-pytorch")
            #    sys.path.insert(0, '/content/stylegan3-Expl0dingCat-pytorch')

            import dnnlib
            import legacy

            if True:
                import numpy as np
                import pyformulas as pf

                import pandas as pd
                import numpy as np

                sg3_latents = np.random.rand((1), self._G3ms[0].z_dim)
                vol = 1

                base_latents = sg3_latents  # .detach().clone()
                #            cons_latents = base_latents
                cons_latents_flatten = base_latents.reshape(len(base_latents[0]))

                cons_index = 0
                while cons_index < len(cons_latents_flatten):
                    if self.ports is not None:
                        for name, port in self.ports.items():
                            if port.data is not None:
                                for (colname, colval) in port.data.items():
                                    val = colval.values[len(colval.values) - 1]
                                    if cons_index < len(cons_latents_flatten):
                                        cons_latents_flatten[cons_index] = val - 0.5
                                        cons_index = cons_index + 1
                cons_latents = cons_latents_flatten.reshape(1, len(base_latents[0]))

                #               device = torch.device('cuda')

                #        if hasattr(G.synthesis, 'input'):
                #            m = make_transform(translate, rotate)
                #            m = np.linalg.inv(m)
                #            G.synthesis.input.transform.copy_(torch.from_numpy(m))

                #                z = psd_array_sg2 * vol
                #                seed=1
                z = torch.from_numpy(cons_latents).to(self._device)
                #                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G3m.z_dim)).to(device)
                truncation_psi = 1
                #                truncation_psi=0.5
                #                noise_mode='const'
                #                noise_mode='random'
                noise_mode = "none"
                label = torch.zeros([1, self._G3ms[0].c_dim], device=self._device)
                # if G3m.c_dim != 0:
                #    label[:, class_idx] = 1

                img = self._G3ms[0](
                    z, label, truncation_psi=truncation_psi, noise_mode=noise_mode
                )
                img = (
                    (img.permute(0, 2, 3, 1) * 127.5 + 128)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                #                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

                images = [img[0].cpu().numpy()]

                #                z_samples = psd_array_sg2 * vol
                #                w_samples = G3m.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
                #                w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
                #                w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
                #                w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
                #                ws3m = (w_opt).repeat([1, G3m.mapping.num_ws, 1])

                #                synth_images = G3m.synthesis(ws3m, noise_mode='const')
                #                synth_images = (synth_images + 1) * (255/2)
                #                synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                # out.append_data(synth_images)
                #                images=[synth_images]

                if True:

                    xsize = 1024
                    ysize = 1024
                    #                  xsize=512
                    #                  ysize=512

                    image_pil = PIL.Image.fromarray(images[0], "RGB")
                    # if generate&gen_sg2_shawwn:
                    #  display(image_pil)
                    # print(image_pil)
                    image_asarray = np.asarray(image_pil)
                    # print(image_asarray)
                    #                  time1111=perf_counter()
                    # print (f'1111: {(time1111-time000):.1f}s')
                    # global video_out
                    # video_out.append_data(image_asarray)
                    #                  time1112=perf_counter()
                    # print (f'1112: {(time1112-time000):.1f}s')
                    img = image_pil.resize((xsize, ysize), PIL.Image.Resampling.LANCZOS)
                    # print(img)
                    #                  time1113=perf_counter()
                    # print (f'1113: {(time1113-time000):.1f}s')
                    #                  buffer = BytesIO()
                    #                  if generate&gen_jpeg:
                    #                    img.save(buffer,format="JPEG")                  #Enregistre l'image dans le buffer
                    #                  if generate&gen_png:
                    #                  img.save(buffer,format="PNG")                  #Enregistre l'image dans le buffer
                    # img.save('/content/gdrive/MyDrive/EEG-GAN-audio-video/out/'+
                    #          f'{(time100*1000):9.0f}'+'/'+f'{(time000*1000):9.0f}'+'.png',format="PNG")

                    #                  buffer.seek(0)
                    #                 time1114=perf_counter()
                    # print (f'1114: {(time1114-time000):.1f}s')
                    #                  myimage = buffer.getvalue()

                    #                  if draw_fps:

                    #                    time111=perf_counter()
                    #                    draw_fps=f'fps: {1/(time111-time001):3.2f}'
                    # print (f'fps: {1/(time111-time001):.1f}s')
                    # print (f'111-001: {(time111-time001):.1f}s')

                    #                    draw = ImageDraw.Draw(img)
                    #                    draw.text((0, 0), draw_fps, font=font, fill='rgb(0, 0, 0)', stroke_fill='rgb(255, 255, 255)', stroke_width=1)
                    #                    img = draw._image
                    #                    time001=time111

                    image = np.asarray(img)
                    image = image[:, :, ::-1]
            if self._to_screen:
                self._screen.update(image)


#            video_outs[shows_idx].append_data(image)
