"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

class ApplyInverseEpochs(Node):

  """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
  """

  def __init__(self, ch_names_pick, duration, overlap, sfreq, to_screen=False, cache_fwd=True, fname_fwd=None, raw_fname=None, from_bdf=None, epochs_con=10, epochs_inverse_con=1, epochs_inverse_cov=165, inverse_subject='fsaverage', inverse_snr=1.0, inverse_method='dSPM', inverse_parc='HCPMMP1', inverse_standard_montage='standard_1005', gamepad_inverse_peaks_labels0 = None, gamepad_inverse_peaks_labels1 = None, gamepad_inverse_peaks_label = None):
   """
        Args:
            value (int): The value to add to each cell.
   """
#        self._value = value
   if True:
        self._to_screen = to_screen
        self._cache_fwd = cache_fwd
        self._fname_fwd = fname_fwd
        self._from_bdf = from_bdf
        self._raw_fname = raw_fname
        self._epochs_con = epochs_con
        self._epochs_inverse_con = epochs_inverse_con
        self._epochs_inverse_cov = epochs_inverse_cov
        self._inverse_subject = inverse_subject
        self._inverse_snr = inverse_snr
        self._inverse_method = inverse_method
        self._inverse_parc = inverse_parc
        self._inverse_standard_montage = inverse_standard_montage
        self._gamepad_inverse_peaks_labels0 = gamepad_inverse_peaks_labels0
        self._gamepad_inverse_peaks_labels1 = gamepad_inverse_peaks_labels1
        self._gamepad_inverse_peaks_label = gamepad_inverse_peaks_label


        self._ch_names_pick = ch_names_pick
        self._duration = duration
        self._overlap = overlap
        self._sfreq = sfreq
        


   if True:
    import mne
    from mne import io
    from mne.datasets import sample
    from mne.minimum_norm import read_inverse_operator, compute_source_psd

#    from mne.connectivity import spectral_connectivity, seed_target_indices

    import pandas as pd
    import numpy as np      






















   if True:
       # -*- coding: utf-8 -*-
       """
       .. _tut-eeg-fsaverage-source-modeling:

       ========================================
       EEG forward operator with a template MRI
       ========================================

       This tutorial explains how to compute the forward operator from EEG data
       using the standard template MRI subject ``fsaverage``.

       .. caution:: Source reconstruction without an individual T1 MRI from the
             subject will be less accurate. Do not over interpret activity
             locations which can be off by multiple centimeters.

       Adult template MRI (fsaverage)
       ------------------------------
       First we show how ``fsaverage`` can be used as a surrogate subject.
       """

       # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
       #          Joan Massich <mailsik@gmail.com>
       #          Eric Larson <larson.eric.d@gmail.com>
       #
       # License: BSD-3-Clause

       import os.path as op
       import numpy as np

       import mne
       from mne.datasets import eegbci
       from mne.datasets import fetch_fsaverage

       # Download fsaverage files
       fs_dir = fetch_fsaverage(verbose=True)
       self._subjects_dir = op.dirname(fs_dir)

       if True:
#       if False:
         # The files live in:
         self._subject = self._inverse_subject
         self._trans = self._inverse_subject  # MNE has a built-in fsaverage transformation
         self._src = op.join(fs_dir, 'bem', self._subject+'-ico-5-src.fif')
         self._bem = op.join(fs_dir, 'bem', self._subject+'-5120-5120-5120-bem-sol.fif')
#         subject = 'fsaverage'
#         trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
#         src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
#         bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
       

       ##############################################################################
       # Load the data
       # ^^^^^^^^^^^^^
       #
       # We use here EEG data from the BCI dataset.
       #
       # .. note:: See :ref:`plot_montage` to view all the standard EEG montages
       #           available in MNE-Python.

       if (self._raw_fname is None) and (self._from_bdf is None):
         self._raw_fname, = eegbci.load_data(subject=1, runs=[6])
       else:
         if not(self._raw_fname is None):
           self._raw_fname = self._raw_fname
         if not(self._from_bdf is None):
           self._raw_fname = self._from_bdf
       import pathlib

       if (pathlib.Path(self._raw_fname).suffix=='.bdf'):
         self._raw = mne.io.read_raw_bdf(self._raw_fname, preload=True)
       if (pathlib.Path(self._raw_fname).suffix=='.edf'):
         self._raw = mne.io.read_raw_edf(self._raw_fname, preload=True)

       # Clean channel names to be able to use a standard 1005 montage
       new_names = dict(
           (ch_name,
            ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
           for ch_name in self._raw.ch_names)
       self._raw.rename_channels(new_names)

       # Read and set the EEG electrode locations, which are already in fsaverage's
       # space (MNI space) for standard_1020:

       ch_names = self._ch_names_pick
       self._raw_bdf = self._raw
       self._raw = self._raw.pick(ch_names)
       
       if True:
#       if False:
         self._montage = mne.channels.make_standard_montage(self._inverse_standard_montage)
#       montage = mne.channels.make_standard_montage('standard_1005')
#       montage = mne.channels.make_standard_montage('biosemi32')

       
         self._raw.set_montage(self._montage)
         self._raw.set_eeg_reference(projection=True)  # needed for inverse modeling

       # Check that the locations of EEG electrodes is correct with respect to MRI
#       mne.viz.plot_alignment(
#           raw.info, src=src, eeg=['original', 'projected'], trans=trans,
#           show_axes=True, mri_fiducials=True, dig='fiducials')

       ##############################################################################
       # Setup source space and compute forward
       # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

       if True:
#       if False:
#       if not (FLAGS.fname_fwd is None):
         if self._cache_fwd:
           if (self._fname_fwd is None):
             from pathlib import Path
             self._fname_fwd = 'inverse_'+Path(self._raw_fname).stem+'_fwd.fif'
           else:
             self._fname_fwd = self._fname_fwd
           import os.path
           if os.path.isfile(self._fname_fwd):
             self._fwd = mne.read_forward_solution(self._fname_fwd)
           else:
             self._fwd = mne.make_forward_solution(self._raw.info, trans=self._trans, src=self._src,
                                         bem=self._bem, eeg=True, mindist=5.0, n_jobs=None)
             mne.write_forward_solution(self._fname_fwd, self._fwd)
         else:
           self._fwd = mne.make_forward_solution(self._raw.info, trans=self._trans, src=self._src,
                                       bem=self._bem, eeg=True, mindist=5.0, n_jobs=None)
         print(self._fwd)

       ##############################################################################
       # From here on, standard inverse imaging methods can be used!
       #
       # Infant MRI surrogates
       # ---------------------
       # We don't have a sample infant dataset for MNE, so let's fake a 10-20 one:

#       ch_names_ = \
#           'Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2'.split()
#       ch_names_=ch_names_pick
#       data = np.random.RandomState(0).randn(len(ch_names_), 1000)
#       info = mne.create_info(ch_names_, 1000., 'eeg')
#       raw = mne.io.RawArray(data, info)

       self._mon=self._montage
#       trans = mne.channels.compute_native_head_t(mon)

       raw=None
   
   
   
   
  
   if True:
              # Compute inverse solution and for each epoch
#              snr = 1.0           # use smaller SNR for raw data
#              inv_method = 'dSPM'
#              parc = 'aparc.a2009s'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
              self._snr = self._inverse_snr
              self._inv_method = self._inverse_method
              self._parc = self._inverse_parc
#              parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

              self._lambda2 = 1.0 / self._snr ** 2

              # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
              print('subject:',self._subject)
#              subject = 'fsaverage'
              self._labels_parc = mne.read_labels_from_annot(self._subject, parc=self._parc,
                                                       subjects_dir=self._subjects_dir)
#              print('labels_parc:',labels_parc)
              remove_unknown_label = True
              while remove_unknown_label:
                remove_unknown_label = False
                for label in self._labels_parc:
                  if label.name.startswith('unknown') or label.name.startswith('???'):
                    self._labels_parc.remove(label)
                    remove_unknown_label = True
#              print('labels_parc:',labels_parc)

              if not (self._gamepad_inverse_peaks_labels0 is None):
                print(self._gamepad_inverse_peaks_labels0)
                self._gamepad_inverse_peaks_indices0 = []
                for idx0 in range(len(self._labels_parc)):
                  for idx1 in range(len(self._gamepad_inverse_peaks_labels0)):
                    if self._labels_parc[idx0].name == self._gamepad_inverse_peaks_labels0[idx1]:
                      self._gamepad_inverse_peaks_indices0.append(idx0)
                print(self._gamepad_inverse_peaks_indices0)
                if (self._gamepad_inverse_peaks_labels1 is None):
                  self._gamepad_inverse_peaks_indices1 = self._gamepad_inverse_peaks_indices0
              if not (self._gamepad_inverse_peaks_labels1 is None):
                print(self._gamepad_inverse_peaks_labels1)
                self._gamepad_inverse_peaks_indices1 = []
                for idx0 in range(len(self._labels_parc)):
                  for idx1 in range(len(self._gamepad_inverse_peaks_labels1)):
                    if self._labels_parc[idx0].name == self._gamepad_inverse_peaks_labels1[idx1]:
                      self._gamepad_inverse_peaks_indices1.append(idx0)
                print(self._gamepad_inverse_peaks_indices1)

   if True:
          if not (self._gamepad_inverse_peaks_label is None):  
            self._fname_label_lh = self._subjects_dir + '/' + self._subject + '/label/lh.'+self._gamepad_inverse_peaks_label+'.label'
#            fname_label_lh = subjects_dir + '/' + subject + '/label/lh.aparc.label'
            self._label_lh = mne.read_label(self._fname_label_lh)
#            fname_label_rh = subjects_dir + '/' + subject + '/label/rh.aparc.label'
            self._fname_label_rh = self._subjects_dir + '/' + self._subject + '/label/rh.'+self._gamepad_inverse_peaks_label+'.label'
            self._label_rh = mne.read_label(self._fname_label_rh)
            self._label = self._label_lh + self._label_rh
          else:
            self._label = None



   if True:
                self._labels_parc_names = []
                for label in self._labels_parc:
                  self._labels_parc_names.append(label.name)



   if True:
     self._inv = None


































   if True:
          import numpy as np
          import pyformulas as pf 
   if self._to_screen:
          self._canvas = np.zeros((800,800))
          self._screen = pf.screen(self._canvas, 'stylegan3')

  def update(self):
        # Make sure we have a non-empty dataframe
      if self.i.ready():

#        self.o.data = self.i.data.tail(1)

        self.i.data 
            




        import numpy as np
        import matplotlib.pyplot as plt
        import PIL.Image
        from matplotlib.colors import LinearSegmentedColormap
        import mne
        from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
        from mne_connectivity import spectral_connectivity_epochs
        import numpy as np
        from mne.viz import circular_layout
        from mne_connectivity.viz import plot_connectivity_circle
        import matplotlib.pyplot as plt
        
        raw_data = self.i.data[self._ch_names_pick].to_numpy().T
#        raw_data = raw_data[:,-(len(raw_data[0])-2):]

        mne.set_log_level('CRITICAL')
        ch_types_pick = ['eeg'] * len(self._ch_names_pick)
        info_pick = mne.create_info(ch_names=self._ch_names_pick, sfreq=self._sfreq, ch_types=ch_types_pick)
        raw = mne.io.RawArray(raw_data, info_pick, verbose='ERROR')

        raw.set_montage(self._mon)
        raw.set_eeg_reference(projection=True).apply_proj()
        
        epochs = mne.make_fixed_length_epochs(raw, 
                                            duration=self._duration, preload=True, overlap=self._overlap, 
                                            verbose='ERROR')



        if self._inv is None:
#        if True:
#   if False:
#        if show_inverse_3d or show_inverse_circle_cons:
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmin=0.0, tmax=0.1, n_jobs=10)
#            cov = mne.compute_covariance(epochs[0][ji:ji+75], tmax=0., n_jobs=cuda_jobs, verbose=False)
            
            self._cov = mne.compute_covariance(epochs[-self._epochs_inverse_cov:], n_jobs=1, verbose='CRITICAL')
#              cov = mne.compute_covariance(epochs[0][:-epochs_inverse_cov], tmax=0., n_jobs=cuda_jobs, verbose='CRITICAL')
#            else:
#              cov = mne.compute_covariance(epochs[ji:ji+epochs_inverse_cov], tmax=0., n_jobs=cuda_jobs, verbose='CRITICAL')
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmin=0.0, tmax=0.1, n_jobs=10)
#            cov = mne.compute_covariance(epochs[0][ji:ji+10], tmax=0., n_jobs=10)
#     cov = mne.compute_covariance(epochs, tmax=0.)
            self._evoked = epochs[len(epochs)-1].average()  # trigger 1 in auditory/left
#            evoked = epochs[0][ji].average()  # trigger 1 in auditory/left
#            evoked.plot_joint()
   
            self._inv = mne.minimum_norm.make_inverse_operator(
                  self._evoked.info, self._fwd, self._cov, 
                  verbose=False, 
#                  verbose=True, 
                  depth=None, fixed=False)

            self._src = self._inv['src']

        if True:
            stcs = apply_inverse_epochs(
#                    epochs[0][ji:ji+1], 
#                    epochs[0][ji:ji+n_jobs],
#                    epochs,
                    epochs[-self._epochs_inverse_con:],
                    self._inv, self._lambda2, self._inv_method,
                                          pick_ori=None, return_generator=True, verbose='CRITICAL')

              # Average the source estimates within each label of the cortical parcellation
              # and each sub-structure contained in the source space.
              # When mode = 'mean_flip', this option is used only for the cortical labels.
            
            label_ts = mne.extract_label_time_course(
                  stcs, self._labels_parc, self._src, mode='mean_flip', 
                  allow_empty=False,
#                  allow_empty=True,
                  return_generator=False, 
#                  return_generator=True, 
#                  verbose=True
                  verbose=False
                  )            
#            print(inv)
#            print(stcs)
#            print(src)
#            print(label_ts)
            
            
#            print(raw)
#            print(epochs)
            
#            print(self.i.data.index)
            
            
            

#        data = pd.DataFrame(label_ts[0], columns = self._labels_parc, index = [self.i.data.index[-self._epochs_inverse_con:]])
        data = pd.DataFrame(label_ts[0].T, columns = self._labels_parc_names, index = self.i.data.index[-len(label_ts[0].T):])
#        data = pd.DataFrame(con_tril, index = self._con_tril_names_2)
#        data.index = pd.MultiIndex.from_tuples(data.index, names=['0-1','1-0'])
#        data = data.T

        self.o.data = data
            


#        if self._to_screen:
#                  self._screen.update(image)




#            video_outs[shows_idx].append_data(image)


