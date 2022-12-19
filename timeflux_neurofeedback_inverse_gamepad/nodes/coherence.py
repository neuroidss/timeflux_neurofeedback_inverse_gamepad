"""Simple example nodes"""

from timeflux.core.node import Node

import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

class SpectralConnectivityEpochs(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self, ch_names_pick, epochs_con, method, fmin, fmax, n_jobs, duration, overlap, sfreq):
        """
        Args:
            value (int): The value to add to each cell.
        """
#        self._value = value
        
        self._ch_names_pick = ch_names_pick
        self._epochs_con = epochs_con
        self._method = method
        self._fmin = fmin
        self._fmax = fmax
        self._n_jobs = n_jobs
        self._duration = duration
        self._overlap = overlap
        self._sfreq = sfreq
        
        cons_len=int(len(self._ch_names_pick)*(len(self._ch_names_pick)-1)/2)
#        fs_mult=3
#        audio_volume_mult=200
#  cons_dur=fs_mult#fps
#        cons_dur=int(fps*10)
#        audio_cons_fs=int(cons_len*(fs_mult-0.0))
#        cons_index=0
#        cons=np.zeros((cons_dur,cons_len),dtype=float)


        self._cohs_tril_indices=np.zeros((2,cons_len),dtype=int)
        cohs_tril_indices_count=0
        for cons_index_diag in range(len(self._ch_names_pick)):
          for cons_index_diag_2 in range(2):
            for cons_index_diag_r in range(cons_index_diag+1):
              cons_index_diag_r_i=(cons_index_diag-cons_index_diag_r)
              if cons_index_diag+cons_index_diag_r+cons_index_diag_2+1<len(self._ch_names_pick):
                if cohs_tril_indices_count<cons_len:
                  self._cohs_tril_indices[0][cohs_tril_indices_count]=cons_index_diag+cons_index_diag_r+cons_index_diag_2+1
                  self._cohs_tril_indices[1][cohs_tril_indices_count]=cons_index_diag_r_i
                  cohs_tril_indices_count=cohs_tril_indices_count+1
        self._con_tril_names = []
        self._con_tril_names_2 = []
        for idx in range(len(self._cohs_tril_indices[0])):
          self._con_tril_names.append(self._ch_names_pick[self._cohs_tril_indices[0][idx]]+'-'+self._ch_names_pick[self._cohs_tril_indices[1][idx]])
          self._con_tril_names_2.append((self._ch_names_pick[self._cohs_tril_indices[0][idx]]+'-'+self._ch_names_pick[self._cohs_tril_indices[1][idx]], self._ch_names_pick[self._cohs_tril_indices[1][idx]]+'-'+self._ch_names_pick[self._cohs_tril_indices[0][idx]]))
        

    def update(self):
        # Make sure we have a non-empty dataframe
      if self.i.ready():

        self.o.meta = self.i.meta
        self.o.data = self.i.data.tail(1)

        raw_data = self.i.data[self._ch_names_pick].to_numpy().T

        mne.set_log_level('CRITICAL')
        
        ch_types_pick = ['eeg'] * len(self._ch_names_pick)
        info_pick = mne.create_info(ch_names=self._ch_names_pick, sfreq=self._sfreq, ch_types=ch_types_pick)
        raw = mne.io.RawArray(raw_data, info_pick, verbose='ERROR')
#        raw.set_montage(mon)
#        raw = raw.pick(ch_names_pick)
#        print('raw: ', raw)

        raw.load_data()

        epochs = mne.make_fixed_length_epochs(raw, 
                                            duration=self._duration, preload=True, overlap=self._overlap, 
                                            verbose='ERROR')
        

#        for (colname,colval) in self.i.data.items():
#                  print('colname, colval.values:',colname, colval.values)
#          if np.max(colval.values)-np.min(colval.values) == 0:
#                    val = np.nan
#          else:
#                    val = (colval.values[len(colval.values)-1]-np.min(colval.values))/(np.max(colval.values)-np.min(colval.values))
#                  print('self.o.data:',self.o.data)
#                  print('self.o.data.iloc[0].at[colname]:',self.o.data.iloc[0].at[colname])
#                    self.o.data.iloc[0].at[colname] = val

        indices=None
#              print('label_ts:',label_ts)
        con = spectral_connectivity_epochs(
                  epochs[0:0+self._epochs_con], indices=indices, method=self._method, mode='multitaper', sfreq=self._sfreq, fmin=self._fmin,
                  fmax=self._fmax, faverage=True, mt_adaptive=True, n_jobs=self._n_jobs, verbose='CRITICAL')
                  
#                              cons=np.roll(cons,1,axis=0)
        conmat = con.get_data(output='dense')[:, :, 0]
#          print(conmat.shape)
#          cons[1:,:] = cons[:len(cons),:]
        con_tril=conmat[(self._cohs_tril_indices[0],self._cohs_tril_indices[1])].flatten('F')

#        print('con_tril:',con_tril)
        data = pd.DataFrame(con_tril, index = self._con_tril_names)
#        data = pd.DataFrame(con_tril, index = self._con_tril_names_2)
#        data.index = pd.MultiIndex.from_tuples(data.index, names=['0-1','1-0'])
        data = data.T
        
#        data.columns = self._con_tril_names
#        data.columns = self._con_tril_names_2
#        data.columns = pd.MultiIndex.from_tuples(self._con_tril_names_2)
#        print('data:',data)
        self.o.data = data
            
            
#                        con_sort=np.sort(np.abs(conmat).ravel())[::-1]
#            con_sort=np.sort(np.abs(con).ravel())[::-1]
#            n_lines=np.argmax(con_sort<vmin)
            
##                        fig,ax = plot_connectivity_circle(conmat, label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names, n_lines=n_lines, 
#            fig,ax = plot_connectivity_circle(con[:, :, 0], label_names,# n_lines=300, 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2), 
##                title=input_fname_name+'_peaks_circle_'+methods[0]+'_'+f'{bands[0][0]:.1f}'+'-'+f'{bands[0][len(bands[0])-1]:.1f}'+'hz_'+'vmin'+str(vmin)+'\n'+f'{ji_fps:.2f}', 
#                title=input_fname_name+'_circle_'+methods[0]+'_'+f'{freqs[0][0]:.1f}'+'-'+f'{freqs[0][len(freqs[0])-1]:.1f}'+'hz_'+'vmin'+str(vmin), 
#               title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+'vmin'+str(vmin)+str(len(epochs[0].events)-2)+'\n'+str(ji), 
#                                             title=input_fname_name+'_circle_'+methods[0]+'_'+str(int(bands[0][0]))+'-'+str(int(bands[0][1]))+'hz_'+str(len(epochs[0].events)-2)+'_'+str(ji),
##                                              node_colors=node_colors,
##                                             show = False, vmin=vmin, vmax=1, fontsize_names=8)#, fig=fig)
#                                             show = False, vmin=0, vmax=1, fontsize_names=8)#16)#, fig=fig)
#         if True:
