"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

class Spectrum(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self, to_screen=False):
        """
        Args:
            value (int): The value to add to each cell.
        """
#        self._value = value
        self._to_screen = to_screen

        if True:

          import numpy as np
          import pyformulas as pf 

        if self._to_screen:

          self._canvas = np.zeros((800,800))
          self._screen = pf.screen(self._canvas, 'spectrum')

        

    def update(self):
        # Make sure we have a non-empty dataframe
      if self.i.ready():

#        self.o.data = self.i.data.tail(1)

        self.i.data 
            
        if True:
            import numpy as np
            import pyformulas as pf 
            import matplotlib.pyplot as plt


            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig = plt.figure(figsize=(800*px, 800*px))

#          plt.imshow(cons, extent=[0,4.2,0,int(32*(32-1)/2)], cmap='jet',
#             vmin=-100, vmax=0, origin='lower', aspect='auto')
            plt.imshow(self.i.data.to_numpy().T, cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)
#            plt.imshow(cons.T[:,::-1], cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)
            plt.colorbar()
#          plt.show()
            plt.close()
          #fig.canvas.draw()

#        if False:
#        if True:
            fig.canvas.draw()

            #image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = np.frombuffer(fig.canvas.tostring_rgb(),'u1')  
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            size1=16*4
            size = 592+size1
#            size = 608
            #im3 = im1.resize((576, 576), Image.ANTIALIAS)
            left=348-int(size/2)+int(size1/2)
            top=404-int(size/2)+int(size1/16)

            image_crop=image[top:top+size,left:left+size]   
            #im2 = im1.crop((left, top, left+size, top+size))

#            image_rot90 = np.rot90(image_crop)
#            image_rot90 = np.rot90(image)

#            screen.update(image)
            image = image_crop
            image = image[:,:,::-1]

            plt.close(fig)
            del fig

        if self._to_screen:
            
            self._screen.update(image)
#            screen.update(image_rot90)




#            video_outs[shows_idx].append_data(image)


class Sound(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self, sound_cons_swap = False):
        """
        Args:
            value (int): The value to add to each cell.
        """
#        self._value = value

        import librosa
        from librosa import load
        from librosa.core import stft, istft
        import numpy as np
        import soundfile as sf
        import sounddevice as sd

        self._sound_cons_swap = sound_cons_swap

        sd.default.reset()

#        fs_mult=3
        self._audio_volume_mult=200
#  cons_dur=fs_mult#fps
#        cons_dur=int(fps*10)
#        audio_cons_fs=int(cons_len*(fs_mult-0.0))


    def update(self):
        # Make sure we have a non-empty dataframe
      if self.i.ready():

#        self.o.data = self.i.data.tail(1)

        self.i.data 
            
        import librosa
        from librosa import load
        from librosa.core import stft, istft
        import numpy as np
        import soundfile as sf
        import sounddevice as sd


        if True:
          import numpy as np
          import pyformulas as pf 
          import matplotlib.pyplot as plt

#          for spectrum_db_range in range(spectrum_db):
#           spectrum_db=cons[cons_index] 

#          spectrum_db=np.abs(cons.T[:,::-1])
          fs_mult=3
          audio_cons_fs=int(len(self.i.data.to_numpy())*(fs_mult-0.0))

          spectrum_db=np.abs(self.i.data.to_numpy().T)
          print(self.i.data.to_numpy().shape)
          spectrum_db_l=spectrum_db[:int(len(spectrum_db)/2)]
          spectrum_db_r=spectrum_db[-int(len(spectrum_db)/2):]
          spectrum_db_r=spectrum_db_r[::-1]
#          spectrum_db_s=[spectrum_db_l,spectrum_db_r]
          spectrum=librosa.db_to_amplitude(spectrum_db)
#          print(spectrum.shape)
#          spectrum = spectrum[:,2:-2]
#          print(spectrum.shape)
          if self._sound_cons_swap:
            spectrum_r=librosa.db_to_amplitude(spectrum_db_l)
            spectrum_l=librosa.db_to_amplitude(spectrum_db_r)
          else:
            spectrum_l=librosa.db_to_amplitude(spectrum_db_l)
            spectrum_r=librosa.db_to_amplitude(spectrum_db_r)
#          back_y = stft.istft(spectrum, 128)
          back_y = istft(spectrum)*self._audio_volume_mult
          back_y_l = istft(spectrum_l)
          back_y_r = istft(spectrum_r)
          back_y_s = np.asarray([back_y_l,back_y_r]).T*self._audio_volume_mult

#          sf.write('/content/out/stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24')
#          sf.write('/content/out/file_trim_5s.wav', y, sr, 'PCM_24')
#          sf.write('/content/out/file_trim_5s_back.wav', back_y, sr, 'PCM_24')
#          sr=48000
#          sr=44100
#          sr=22050
          #sr=11025
#          sr=int(48000/10)
#          sr=int(48000/20)
#          sr=cons_len
          sr=1000
#          sr=4000
#          sr=len(self.i.data.to_numpy().T[0])
#          print(sr)
          sound_cons_buffer_path = ''
          sf.write(sound_cons_buffer_path+'cons_back.wav', back_y, sr, 'PCM_24')
          sf.write(sound_cons_buffer_path+'cons_back_s.wav', back_y_s, sr, 'PCM_24')
          filename=sound_cons_buffer_path+'cons_back_s.wav'
#          device=
          #print(sd.query_devices())
#          try:
#          if False:
          if True:
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data, fs)#, device=device)
          if False:
#          if True:
            data=back_y_s
            fs=audio_cons_fs
            sd.play(data, fs)#, device=device)

            #mydata = sd.rec(int(data),fs,channels=2, blocking=True)
            #sf.write(filename, data, fs)


            #status = sd.wait()
#          except KeyboardInterrupt:
#            parser.exit('\nInterrupted by user')
#          except Exception as e:
#            parser.exit(type(e).__name__ + ': ' + str(e))
#          if status:
#            parser.exit('Error during playback: ' + str(status))          

          #from librosa import output
          #librosa.output.write_wav('/content/out/file_trim_5s.wav', y, s_r)
#          librosa.output.write_wav('/content/out/file_trim_5s_back.wav', back_y, sample_rate)
             

#            fig.show()
#            fig.show(0)

#            draw() 
#print 'continuing computation'
#            show()


#            video_outs[shows_idx].append_data(image)


