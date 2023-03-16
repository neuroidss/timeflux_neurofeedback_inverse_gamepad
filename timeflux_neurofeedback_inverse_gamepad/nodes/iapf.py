"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
#from mne_connectivity import spectral_connectivity_epochs

# General imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectra


class IAPF(Node):

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
        peak_width_limits=[1, 6], 
        min_peak_height=0.15,
        peak_threshold=2., 
        max_n_peaks=1,
        nan_policy = 'mean',

        epochs_con=1,
        n_jobs=1,
        to_screen=False,
        vmin=0,
        con_name="neurofeedback",
        node_colors=None,
        xsize=1500,
        ysize=1500,
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value

        self._peak_width_limits=peak_width_limits
        self._min_peak_height=min_peak_height
        self._peak_threshold=peak_threshold
        self._max_n_peaks=max_n_peaks
        self._nan_policy = nan_policy
        
        self._ch_names_pick = ch_names_pick
        self._epochs_con = epochs_con
        self._method = method
        self._fmin = fmin
        self._fmax = fmax
        self._n_jobs = n_jobs
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

        self._cohs_tril_indices = None

        if self._to_screen:
            import numpy as np
            import pyformulas as pf

            self._canvas = np.zeros((800, 800))
            self._screen = pf.screen(self._canvas, "iapf")

    def check_nans(self, data, nan_policy='zero'):
        """Check an array for nan values, and replace, based on policy."""

        # Find where there are nan values in the data
        nan_inds = np.where(np.isnan(data))

        # Apply desired nan policy to data
        if nan_policy == 'zero':
            data[nan_inds] = 0
        elif nan_policy == 'mean':
            data[nan_inds] = np.nanmean(data)
        else:
            raise ValueError('Nan policy not understood.')

        return data


    def update(self):
        # Make sure we have a non-empty dataframe
        if self.i.ready():

            import numpy as np
            self.o.meta = self.i.meta
            self.o.data = self.i.data.tail(1)


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

            spectrum = raw.compute_psd(method="welch", 
                fmin=self._fmin, 
                fmax=self._fmax, 
                tmin=0, 
                tmax=self._duration,#TODO:check
#                n_overlap=self._overlap, 
#                n_fft=int(self._sfreq)*32,
                n_fft=len(raw),
                verbose="ERROR",
            )
            spectra = spectrum.get_data()
            freqs = spectrum.freqs
            
            fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose="ERROR")

            # Define the frequency range to fit
            freq_range = [self._fmin, self._fmax]
            
            fg.fit(freqs, spectra, freq_range)
            
            # Extract alpha peaks
            alphas = get_band_peak_fg(fg, freq_range)

            # Extract the power values from the detected peaks
            alpha_pw = alphas[:, 1]


            exps = fg.get_params('aperiodic_params', 'exponent')
#            spectra = fg.get_fooof(np.argmax(exps)).power_spectrum
            band_power = self.check_nans(get_band_peak_fg(fg, freq_range)[:, 1])
            spectra = fg.get_fooof(np.argmax(band_power)).power_spectrum
            freqs = fg.freqs
#            print(fg.freqs)
            print(alphas)
            print(alpha_pw)
#            print(freqs.shape)
#            print(spectra.shape)
#            print(freqs)
#            print(spectra)

            data = pd.DataFrame(
                spectra,
                index=freqs,
                columns=[self.i.data.index[self.i.data.index.size - 1]],
            )
            #        data = pd.DataFrame(con_tril, index = self._con_tril_names_2)
            #        data.index = pd.MultiIndex.from_tuples(data.index, names=['0-1','1-0'])
            data = data.T
            #        data.index[0] = self.i.data.index[0]

            #        data.columns = self._con_tril_names
            #        data.columns = self._con_tril_names_2
            #        data.columns = pd.MultiIndex.from_tuples(self._con_tril_names_2)
            #        print('data:',data)
            self.o.data = data

            if self._to_screen:
                import numpy as np
                import pyformulas as pf
                from mne_connectivity.viz import plot_connectivity_circle
                import matplotlib.pyplot as plt
                

                px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                fig, ax = plt.subplots(figsize=(self._xsize*px, self._ysize*px))
#                fig, ax = plt.subplots(figsize=(self._xsize*px, self._ysize*px), facecolor='black',
#                       subplot_kw=dict(polar=True))

                plt.ylim([-2, 2])
                ax.plot(freqs,spectra)
                ax.plot(alphas[0][0],alphas[0][1], marker="o", ls="", ms=3)
#                fg.plot(fig)
#                plot_hist(fg.get_params('peak_params', 0)[:, 0], 'Center Frequency',
#                  'Peaks - Center Frequencies', x_lims=fg.freq_range, ax=ax)


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

#                rotate = True
#                if rotate:
##                    image_rot90 = np.rot90(image_crop)
#                    image_rot90 = np.rot90(image)
#                    image = image_rot90
#                #              screen.update(image_rot90)
#                else:
##                    image = image_crop
#                    image = image
                #            image_rot90 = np.rot90(image)

                #            screen.update(image)
                #              screen.update(image_crop)

                ##            image = image[:,:,::-1]
                ##            screen.update(image)

                plt.close(fig)
                del fig

                image = image[:, :, ::-1]
                self._screen.update(image)


#            video_outs[shows_idx].append_data(image)
