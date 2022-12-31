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



