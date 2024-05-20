"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs


class SpectralConnectivityEpochs(Node):

    """
        Parameters
        ----------
        ch_names_pick: list
            channels names to pick
        epochs_con: int
            number of epochs for connectivity
        method: string
            connectivity method one of 'coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased'
        fmin: float
            frequency minimum
        fmax: float
            frequency maximum
        n_jobs: int
            number of jobs
        duration: float
            duration in seconds
        overlap: float
            overlap in seconds
        sfreq: int
            number of samples per second
        to_screen: boolean
            output to screen
            Default: False
        vmin: float
            duration in seconds
            Default: 0
        con_name: string
            connectivity name
            Default: 'neurofeedback'
        node_colors: list
            node colors
            Default: None
    """

    def __init__(
        self,
        ch_names_pick,
        method,
        fmin,
        fmax,
        duration,
        overlap,
        sfreq,
        epochs_con=1,
        n_jobs=1,
        to_screen=False,
        vmin=0,
        con_name="neurofeedback",
        node_colors=None,
        xsize=1500,
        ysize=1500,
        triangle=True,
        tri=None,#'tril','triu',None,'both'
        to_file=None,
        video_filename=None,
        video_path="/tmp",
        to_video=False,
        to_bdf=False,
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value

#        import objgraph

        self._ch_names_pick = ch_names_pick
        self._epochs_con = epochs_con
        self._method = method
        self._fmin = fmin
        self._fmax = fmax
        self._n_jobs = n_jobs
        self._duration_init = duration
        self._duration = duration
        self._overlap = overlap
        self._sfreq = sfreq
        self._to_screen = to_screen
        self._vmin = vmin
        self._con_name = con_name
        self._node_colors = node_colors
        self._index = None
        self._xsize = xsize
        self._ysize = ysize
        self._triangle = triangle
        self._tri = tri
        self._to_file = to_file
        self._to_video = to_video
        self._to_bdf = to_bdf

        self._cohs_tril_indices = None

        if self._to_screen or self._to_video:
            import numpy as np
            import pyformulas as pf

            self._canvas = np.zeros((self._xsize, self._ysize))
            if self._to_screen:
                self._screen = pf.screen(self._canvas, con_name)
            
            if self._to_video:
                import imageio
#                self._fps=1
                self._fps=1/(duration - overlap)
                import os
                import time
                os.makedirs(video_path, exist_ok=True)
                now = time.gmtime()
                if video_filename is None:
                    video_filename = os.path.join(
                        video_path,
                        time.strftime("%Y%m%d-%H%M%S.mp4", now),
                    )
                else:
                    video_filename = os.path.join(video_path, video_filename)
                from datetime import datetime
                now = datetime.now()
                self.logger.info("Saving to %s", video_filename)

                self._out = imageio.get_writer(video_filename, fps=self._fps)


    def update(self):
        # Make sure we have a non-empty dataframe
        if self.i.ready():
#            import objgraph
            import numpy as np
#            print('len(self.i.data): ', len(self.i.data))

            self.o.meta = self.i.meta
#            self.o.data = self.i.data
            if self._to_bdf:
                coherence_repeat = int(self._sfreq*(self._duration - self._overlap))
                self.o.data = self.i.data.tail(coherence_repeat)
            else:
                self.o.data = self.i.data.tail(1)
            

            if self._cohs_tril_indices is None:
              if self._ch_names_pick is None:
                    self._ch_names_pick = list(self.i.data.columns)

              if self._triangle:
               if self._tri is None:
                cons_len = int(
                    len(self._ch_names_pick) * (len(self._ch_names_pick) - 1) / 2
                )
                #        fs_mult=3
                #        audio_volume_mult=200
                #  cons_dur=fs_mult#fps
                #        cons_dur=int(fps*10)
                #        audio_cons_fs=int(cons_len*(fs_mult-0.0))
                #        cons_index=0
                #        cons=np.zeros((cons_dur,cons_len),dtype=float)

                self._cohs_tril_indices = np.zeros((2, cons_len), dtype=int)
                cohs_tril_indices_count = 0
                for cons_index_diag in range(len(self._ch_names_pick)):
                    for cons_index_diag_2 in range(2):
                        for cons_index_diag_r in range(cons_index_diag + 1):
                            cons_index_diag_r_i = cons_index_diag - cons_index_diag_r
                            if (
                                cons_index_diag
                                + cons_index_diag_r
                                + cons_index_diag_2
                                + 1
                                < len(self._ch_names_pick)
                            ):
                                if cohs_tril_indices_count < cons_len:
                                    self._cohs_tril_indices[0][
                                        cohs_tril_indices_count
                                    ] = (
                                        cons_index_diag
                                        + cons_index_diag_r
                                        + cons_index_diag_2
                                        + 1
                                    )
                                    self._cohs_tril_indices[1][
                                        cohs_tril_indices_count
                                    ] = cons_index_diag_r_i
                                    cohs_tril_indices_count = (
                                        cohs_tril_indices_count + 1
                                    )
               if self._tri == 'tril':
                self._cohs_tril_indices = np.tril_indices(len(self._ch_names_pick), k = -1)
               if self._tri == 'triu':
                self._cohs_tril_indices = np.triu_indices(len(self._ch_names_pick), k = 1)
              else:
               if self._tri is None:
                cons_len = int(
                    len(self._ch_names_pick) * (len(self._ch_names_pick))
                )
                #        fs_mult=3
                #        audio_volume_mult=200
                #  cons_dur=fs_mult#fps
                #        cons_dur=int(fps*10)
                #        audio_cons_fs=int(cons_len*(fs_mult-0.0))
                #        cons_index=0
                #        cons=np.zeros((cons_dur,cons_len),dtype=float)

                import numpy as np

                self._cohs_tril_indices = np.zeros((2, cons_len), dtype=int)
                cohs_tril_indices_count = 0
                for cons_index_diag in range(len(self._ch_names_pick)):
                    for cons_index_diag_2 in range(len(self._ch_names_pick)):
                                    self._cohs_tril_indices[0][
                                        cohs_tril_indices_count
                                    ] = (
                                        cons_index_diag
                                    )
                                    self._cohs_tril_indices[1][
                                        cohs_tril_indices_count
                                    ] = cons_index_diag_2
                                    cohs_tril_indices_count = (
                                        cohs_tril_indices_count + 1
                                    )
               if self._tri == 'both':
                self._cohs_tril_indices = np.indices((len(self._ch_names_pick), len(self._ch_names_pick)))
                #print(len(self._cohs_tril_indices))
                #print(len(self._cohs_tril_indices[0]))
                #print(len(self._cohs_tril_indices[0][0]))
                self._cohs_tril_indices = np.reshape(self._cohs_tril_indices, (len(self._cohs_tril_indices), len(self._cohs_tril_indices[0])*len(self._cohs_tril_indices[0][0])))
                #print(self._cohs_tril_indices)

            if True:
                self._con_tril_names = []
                self._con_tril_names_2 = []
                for idx in range(len(self._cohs_tril_indices[0])):
                    self._con_tril_names.append(
                        self._ch_names_pick[self._cohs_tril_indices[0][idx]]
                        + "__"
                        + self._ch_names_pick[self._cohs_tril_indices[1][idx]]
                    )
                    self._con_tril_names_2.append(
                        (
                            self._ch_names_pick[self._cohs_tril_indices[0][idx]]
                            + "__"
                            + self._ch_names_pick[self._cohs_tril_indices[1][idx]],
                            self._ch_names_pick[self._cohs_tril_indices[1][idx]]
                            + "__"
                            + self._ch_names_pick[self._cohs_tril_indices[0][idx]],
                        )
                    )

            raw_data = self.i.data[self._ch_names_pick].to_numpy().T

            mne.set_log_level("CRITICAL")

            ch_types_pick = ["eeg"] * len(self._ch_names_pick)
            info_pick = mne.create_info(
                ch_names=self._ch_names_pick, sfreq=self._sfreq, ch_types=ch_types_pick
            )
            raw = mne.io.RawArray(raw_data, info_pick, verbose="ERROR")
            #        raw.set_montage(mon)
            #        raw = raw.pick(ch_names_pick)
            #        print('raw: ', raw)

            raw.load_data()
            if self._duration_init is None:
#              print('raw.n_times: ', raw.n_times)
#              print('raw.times: ', raw.times)
              self._duration = raw.times[len(raw.times)-1]

            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self._duration,
                preload=True,
                overlap=self._overlap,
                verbose="ERROR",
            )

            #        for (colname,colval) in self.i.data.items():
            #                  print('colname, colval.values:',colname, colval.values)
            #          if np.max(colval.values)-np.min(colval.values) == 0:
            #                    val = np.nan
            #          else:
            #                    val = (colval.values[len(colval.values)-1]-np.min(colval.values))/(np.max(colval.values)-np.min(colval.values))
            #                  print('self.o.data:',self.o.data)
            #                  print('self.o.data.iloc[0].at[colname]:',self.o.data.iloc[0].at[colname])
            #                    self.o.data.iloc[0].at[colname] = val

            indices = None
            #              print('label_ts:',label_ts)
            con = spectral_connectivity_epochs(
                epochs[0 : 0 + self._epochs_con],
                indices=indices,
                method=self._method,
                mode="multitaper",
                sfreq=self._sfreq,
                fmin=self._fmin,
                fmax=self._fmax,
                faverage=True,
                mt_adaptive=True,
                n_jobs=self._n_jobs,
                verbose="CRITICAL",
            )

            #                              cons=np.roll(cons,1,axis=0)
            conmat = con.get_data(output="dense")[:, :, 0]
            if not self._triangle:
              for i in range(len(conmat)):
                conmat[i][i] = 1
                for j in range(i, len(conmat)):
                  conmat[i][j] = conmat[j][i]
            #          print(conmat.shape)
            #          cons[1:,:] = cons[:len(cons),:]
            con_tril = conmat[
                (self._cohs_tril_indices[0], self._cohs_tril_indices[1])
              ].flatten("F")

            #        print('con_tril:',con_tril)
            #        data = pd.DataFrame(con_tril, index = self._con_tril_names, columns = [self.i.data.index[0]])
#            data = pd.DataFrame(
#                con_tril,
#                index=self._con_tril_names,
#                columns=[self.i.data.index[self.i.data.index.size - 1]],
#            )
            #        data = pd.DataFrame(con_tril, index = self._con_tril_names_2)
            #        data.index = pd.MultiIndex.from_tuples(data.index, names=['0-1','1-0'])
#            data = data.T
#            print(np.array([con_tril]).shape)
#            print(np.array([con_tril]).T.shape)
            #        data.index[0] = self.i.data.index[0]

            #        data.columns = self._con_tril_names
            #        data.columns = self._con_tril_names_2
            #        data.columns = pd.MultiIndex.from_tuples(self._con_tril_names_2)
            #        print('data:',data)
#            self.o.data = data
#            print(self.o.data)
#            print(data)
            if self._to_bdf:
#                print(coherence_repeat)
                data = pd.DataFrame(
                    np.repeat(np.array([con_tril*1000]), repeats=coherence_repeat, axis=0),
                    index=self.i.data.index[-coherence_repeat:],
                    columns=self._con_tril_names,
                )
#                data = pd.DataFrame(
#                    np.array([con_tril*1000]),
#                    index=[self.i.data.index[self.i.data.index.size - 1]],
#                    columns=self._con_tril_names,
#                )
                self.o.data = pd.concat([self.o.data, data], axis=1)
            else:
                data = pd.DataFrame(
                    np.array([con_tril]),
                    index=[self.i.data.index[self.i.data.index.size - 1]],
                    columns=self._con_tril_names,
                )
                self.o.data = data
#            print(self.o.data)

            if self._to_screen or self._to_video:
                import numpy as np
                import pyformulas as pf
                from mne_connectivity.viz import plot_connectivity_circle
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('QtAgg')
#                matplotlib.use('ipympl')
#                matplotlib.use('GTK3Agg')
#                matplotlib.use('GTK4Agg')
#                matplotlib.use("TkAgg")
#                matplotlib.use('nbAgg')
#                matplotlib.use('WebAgg')
#                matplotlib.use('GTK3Cairo')
#                matplotlib.use('GTK4Cairo')
#                matplotlib.use('wxAgg')

                con_sort = np.sort(np.abs(conmat).ravel())[::-1]
                #            con_sort=np.sort(np.abs(con).ravel())[::-1]
                n_lines = np.argmax(con_sort < self._vmin)

                label_names = self._ch_names_pick
                index = self.i.data.index[0]
                if self._index is None:
                    self._index = index
                index_delta = index - self._index

                px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                fig, ax = plt.subplots(figsize=(self._xsize*px, self._ysize*px), facecolor='black',
                       subplot_kw=dict(polar=True))

                fig, ax = plot_connectivity_circle(
                    conmat,
                    label_names,
                    n_lines=n_lines,
                    title=self._con_name
                    + "_circle_"
                    + self._method
                    + "_"
                    + f"{self._fmin:.1f}"
                    + "-"
                    + f"{self._fmax:.1f}"
                    + "hz_"
                    + "vmin"
                    + str(self._vmin)
                    + "\n"
                    + f"{index_delta.total_seconds():.2f}",
                    show=False,
                    vmin=self._vmin,
                    vmax=1,
                    fontsize_names=8,
                    node_colors=self._node_colors,
                    padding=1.2,
                    ax=ax,
                )  # , fig=fig)

                fig.canvas.draw()

                # image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = np.frombuffer(fig.canvas.tostring_rgb(), "u1")
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                size1 = 16 * 4
                size = 592 + size1
                #            size = 608
                # im3 = im1.resize((576, 576), Image.ANTIALIAS)
                left = 348 - int(size / 2) + int(size1 / 2)
                top = 404 - int(size / 2) + int(size1 / 16)

#                image_crop = image[top : top + size, left : left + size]
                # im2 = im1.crop((left, top, left+size, top+size))

                rotate = True
                if rotate:
#                    image_rot90 = np.rot90(image_crop)
                    image_rot90 = np.rot90(image)
                    image = image_rot90
                #              screen.update(image_rot90)
                else:
#                    image = image_crop
                    image = image
                #            image_rot90 = np.rot90(image)

                #            screen.update(image)
                #              screen.update(image_crop)

                ##            image = image[:,:,::-1]
                ##            screen.update(image)

                if self._to_file is not None:
                  #fig.savefig(self._to_file)
                  from PIL import Image
                  im = Image.fromarray(image)
                  im.save(self._to_file)

#                f = plt.figure()
#                f.clear()
#                plt.close(f)
#                plt.close(fig)
#                del fig
                
                
#                plt.figure().clear()
#                plt.close()


#                plt.clf() 
#                plt.cla()


                image = image[:, :, ::-1]
                if self._to_screen:
                    self._screen.update(image)
                
                if self._to_video:
                    import io
                    img_buf1 = io.BytesIO()
                    from PIL import Image
                    im = Image.fromarray(image)
                    im.save(img_buf1, format='png')
                    img_buf1.seek(0)

                    import imageio
                    im2 = imageio.imread(img_buf1)
                    img_buf1.close()
                    self._out.append_data(im2)

                plt.close()
                import gc
                gc.collect()
#            if False:


#            video_outs[shows_idx].append_data(image)
