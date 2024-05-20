"""timeflux.nodes.hdf5: HDF5 nodes"""

import pandas as pd
import timeflux.helpers.clock as clock
import sys
import os
import time
from timeflux.core.exceptions import WorkerInterrupt, WorkerLoadError
from timeflux.core.node import Node

import pyedflib
import mne
import numpy as np
from datetime import datetime

#    now = datetime.now()
#    dt_string = now.strftime("%Y.%m.%d-%H.%M.%S")
#    output_path=FLAGS.output_path

# Ignore the "object name is not a valid Python identifier" message
import warnings
from tables.exceptions import NaturalNameWarning

warnings.simplefilter("ignore", NaturalNameWarning)


class Replay(Node):
    """Replay a HDF5 file."""

    def __init__(
        self, filename, keys=None, speed=1, timespan=None, resync=True, start=0
    ):
        """
        Initialize.

        Parameters
        ----------
        filename : string
            The path to the HDF5 file.
        keys: list
            The list of keys to replay.
        speed: float
            The speed at which the data must be replayed. 1 means real-time.
            Default: 1
        timespan: float
            The timespan of each chunk, in seconds.
            If not None, will take precedence over the `speed` parameter
            Default: None
        resync: boolean
            If False, timestamps will not be resync'ed to current time
            Default: True
        start: float
            Start directly at the given time offset, in seconds
            Default: 0
        """

        # Load store
        try:
            #            self._store = pd.HDFStore(self._find_path(filename), mode="r")
            mne.set_log_level("CRITICAL")
            #            self._store = pd.HDFStore('store.h5')
            self._bdf = mne.io.read_raw_bdf(
                self._find_path(filename),
                eog=None,
                misc=None,
                stim_channel="auto",
                exclude=(),
                preload=False,
                verbose=None,
            )
            self._bdf.load_data()
            rate = self._bdf.info["sfreq"]
            if keys is None:
                keys = self._bdf.info["ch_names"]
            self._meas_date = pd.Timestamp(self._bdf.info["meas_date"])
            #            print('self._store.info[meas_date]:',self._store.info['meas_date'])
            #            print('self._meas_date:',self._meas_date)
            #            self._meas_date.tz_localize(None)
            #            print('self._meas_date:',self._meas_date)
            self._meas_date = self._meas_date.tz_convert(None)
        #            print('self._meas_date:',self._meas_date)
        #            self._meas_date.tz_localize(None)
        #            print('self._meas_date:',self._meas_date)
        #            print(self._bdf.info)
        #            self._store.pick_channels(keys)
        #            print(self._store.info)
        #            self._store = timeflux.helpers.mne.mne_to_xarray(epochs, context_key, event_id, output='dataarray')
        except IOError as e:
            raise WorkerInterrupt(e)

        # Init
        self._sources = {}
        self._start = pd.Timestamp.max
        #        self._start = self._start.tz_localize(self._meas_date.tz)
        #        self._start.tz_convert(None)
        self._stop = pd.Timestamp.min
        #        self._stop.tz_convert(None)
        #        self._stop = self._stop.tz_localize(self._meas_date.tz)
        self._speed = speed
        self._timespan = None if not timespan else pd.Timedelta(f"{timespan}s")
        self._resync = resync

        for key in keys:
            try:
                # Check format
                #                if not self._store.get_storer(key).is_table:
                #                    self.logger.warning("%s: Fixed format. Will be skipped.", key)
                #                    continue
                # Get first index
                #                first = self._store.select(key, start=0, stop=1).index[0]
                #                print([key])
                #                print(self._store.get_data(picks=[key]))
                #                print(self._store.pick_channels([key]))
                first = self._meas_date + pd.Timedelta(
                    value=self._bdf[0][1][0], unit="seconds"
                )
                #                first = self._meas_date + pd.Timedelta(value=self._store.pick_channels([key])[0][1][0], unit='seconds')
                #                first = self._store.pick_channels([key])[0][1][0]
                #                print(self._store)
                #                print(self._store[0])
                #                print(self._store[0][0])
                #                print(self._store[0,0])
                #                print(self._store[1])
                #                print(self._store[1][0])
                #                print(self._store[1,0])
                #                print(self._store[0])
                #                print(self._store[0][1])
                #                print(self._store[0,1])
                #                first = self._store.pick_channels(key)[0][0]
                # Get last index
                #                nrows = self._store.get_storer(key).nrows
                nrows = len(self._bdf[0][1])
                #                nrows = len(self._store.pick_channels([key])[0][1])
                #                nrows = len(self._store.pick(key)[0])
                #                last = self._store.select(key, start=nrows - 1, stop=nrows).index[0]
                last = self._meas_date + pd.Timedelta(
                    value=self._bdf[0][1][nrows - 1], unit="seconds"
                )
                #                last = self._meas_date + pd.Timedelta(value=self._store.pick_channels([key])[0][1][nrows - 1], unit='seconds')
                #                last = self._store.pick_channels([key])[0][1][nrows - 1]
                #                last = self._store.pick(key)[0][nrows - 1]
                # Check index type
                #                if type(first) != pd.Timestamp:
                #                    self.logger.warning("%s: Invalid index. Will be skipped.", key)
                #                    continue
                # Find lowest and highest indices across stores
                #                print('first:',first)
                #                print('self._start:',self._start)
                if first < self._start:
                    self._start = first
                #                print('last:',last)
                #                print('self._stop:',self._stop)
                if last > self._stop:
                    self._stop = last
                # Extract meta
                #                if self._store.get_node(key)._v_attrs.__contains__("meta"):
                #                    meta = self._store.get_node(key)._v_attrs["meta"]
                #                else:
                #                    meta = {}
                meta = {"rate": rate}
                # Set output port name, port will be created dynamically
                name = "o" + "_" + key.replace("/", "_")
                #                name = "o" + key.replace("/", "_")
                # Update sources
                self._sources[key] = {
                    "start": first,
                    "stop": last,
                    "nrows": nrows,
                    "name": name,
                    "meta": meta,
                }
            except KeyError:
                self.logger.warning("%s: Key not found.", key)

        # Current time
        now = clock.now()

        #        from datetime import datetime

        # use this extension and it adds the timezone
        #        tznow = datetime.now().astimezone()

        # Starting timestamp
        self._start += pd.Timedelta(f"{start}s")
        #        print('self._start:',self._start)

        # Time offset
        self._offset = pd.Timestamp(now) - self._start
        #        self._offset = pd.Timestamp(now,tz=tznow) - self._start
        #        print('self._offset:',self._offset)

        # Current query time
        self._current = self._start
        #        print('self._current:',self._current)

        # Last update
        self._last = now

    #        print('self._last:',self._last)

    def update(self):

        if self._current > self._stop:
            raise WorkerInterrupt("No more data.")

        min = self._current

        if self._timespan:
            max = min + self._timespan
        else:
            now = clock.now()
            ellapsed = now - self._last
            max = min + ellapsed * self._speed
            self._last = now

#        print('self._last:',self._last)

#        print('min,max:',min,max)
#        print('self._sources.items():',self._sources.items())
        for key, source in self._sources.items():
            #           print('key,source:',key,source)

            # Select data
            #            data = self._store.select(key, "index >= min & index < max")
            data = self._bdf.get_data(
                [key],
                tmin=pd.Timedelta(min - self._meas_date).seconds,
                tmax=pd.Timedelta(max - self._meas_date).seconds,
            )
            data = data * 1000000

            # Add offset
            if self._resync:
                data.index += self._offset

            # Update port
            #            print('data:',data)
            data = pd.DataFrame(data.T)
            for idx in range(len(self._bdf.info["ch_names"])):
                data = data.rename(columns={idx: self._bdf.info["ch_names"][idx]})
            #            print('data:',data)
            if data.size > 0:
                #              print('data.shape:',data.shape)
                data_time = np.asarray(range(data.shape[0])) / self._bdf.info["sfreq"]
                data_time = min + pd.to_timedelta(data_time, unit="seconds")
                data.insert(loc=0, column="time", value=data_time)
                data = data.set_index("time")
            #            data.rename(index={1: 'counter'})
            #              print('data.index[0]:',data.index[0])
            #              print('data:',data)
            #            print('source["meta"]:',source["meta"])
            getattr(self, source["name"]).data = data
            getattr(self, source["name"]).meta = source["meta"]

        #        print('self._sources.items():',self._sources.items())

        self._current = max

    def terminate(self):
        self._bdf.close()
#        print('replay bdf closed')

    def _find_path(self, path):
        path = os.path.normpath(path)
        if os.path.isabs(path):
            if os.path.isfile(path):
                return path
        else:
            for base in sys.path:
                full_path = os.path.join(base, path)
                if os.path.isfile(full_path):
                    return full_path
        raise WorkerLoadError(f"File `{path}` could not be found in the search path.")


class Save(Node):
    """Save to HDF5."""

    def __init__(
        self,
        filename=None,
        path="/tmp",
        complib="zlib",
        complevel=3,
        min_itemsize=None,
        sample_rate=None,
        eeg_channels=None,
        pmax=312500,
        pmin=-312500,
        dmax=8388607,
        dmin=-8388608,
        dimension="uV",
        data_key="eeg",
        file_type=pyedflib.FILETYPE_BDFPLUS,
    ):
        """
        Initialize.

        Parameters
        ----------
        filename: string
            Name of the file (inside the path set by parameter). If not set,
            an auto-generated filename is used.
        path : string
            The directory where the HDF5 file will be written.
            Default: "/tmp"
        complib : string
            The compression lib to be used.
            see: https://www.pytables.org/usersguide/libref/helper_classes.html
            Default: "zlib"
        complevel : int
            The compression level. A value of 0 disables compression.
            Default: 3
            see: https://www.pytables.org/usersguide/libref/helper_classes.html
        min_itemsize : int
            The string columns size
            Default: None
            see: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.HDFStore.append.html
            see: http://pandas.pydata.org/pandas-docs/stable/io.html#string-columns

        """
        
        self._filename=filename
        self._path=path
        self._complib=complib
        self._complevel=complevel
        self._min_itemsize=min_itemsize
        self._sample_rate=sample_rate
        self._eeg_channels=eeg_channels
        self._pmax=pmax
        self._pmin=pmin
        self._dmax=dmax
        self._dmin=dmin
        self._dimension=dimension
        self._data_key=data_key
        self._file_type=file_type

        #        self._store = pd.HDFStore(filename, complib=complib, complevel=complevel)
        #        self.min_itemsize = min_itemsize

        #        pmax=312500 #gain 8, vref 2.5 V, 24 bit
        #        dmax = 8388607
        #        dmin = -8388608
        #  if not pmax:
        #      pmax = max(abs(signals.min()), signals.max())
        #        pmin = -pmax

        #        dimension="uV"
        #        data_key="eeg"
        self._bdf = None

    def update(self):
#        print(self)
        if self.i.ready():
#        if self.ports is not None:
            #          bdf.writeSamples(bufs_hstack_cut)
 #           for name, port in self.ports.items():
                port = self.i
#                print(port)
#                print(port.data)
                #                if not name.startswith("i"):
                #                    continue
                #                key = "/" + name[2:].replace("_", "/")
                if port.data is not None:
                    
                  if self._bdf is None:

                        os.makedirs(self._path, exist_ok=True)
                        self._now = time.gmtime()
                        if self._filename is None:
                            self._filename = os.path.join(
                #               path, time.strftime("%Y%m%d-%H%M%S.hdf5", time.gmtime())
                                self._path,
                                time.strftime("%Y%m%d-%H%M%S.bdf", self._now),
                            )
                        else:
                            self._filename = os.path.join(self._path, self._filename)
                        self._now = datetime.now()
                        self.logger.info("Saving to %s", self._filename)
                        if self._sample_rate is None:
#                            for name, port in self.ports.items():
                                print("name,port:", name, port)
                                self._sample_rate = getattr(self, port["name"]).meta["rate"]
        #        rate=512
                        self._rate = self._sample_rate

                        if self._eeg_channels is None:
#                            for name, port in self.ports.items():
#                                print("name,port:", name, port)
                                self._eeg_channels = port.data.columns
        #        print('self:',self)
        #        print('self.ports:',self.ports)
        #        print('self.ports.items():',self.ports.items())
        #        eeg_channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']
                        self._n_channels = len(self._eeg_channels)
        #        file_type = pyedflib.FILETYPE_BDFPLUS  # BDF+
                        self._bdf = pyedflib.EdfWriter(
                            self._filename, n_channels=self._n_channels, file_type=self._file_type
                        )

                        self._headers = []
        # print(ch_names)
                        self._ch_names = self._eeg_channels
                        for channel in self._ch_names:
                            self._headers.append(
                                {
                                    "label": str(channel),
                                    "dimension": self._dimension,
                                    "sample_rate": self._rate,
                                    "physical_min": self._pmin,
                                    "physical_max": self._pmax,
                                    "digital_min": self._dmin,
                                    "digital_max": self._dmax,
                                    "transducer": "",
                                    "prefilter": "",
                                }
                            )
                        self._bdf.setSignalHeaders(self._headers)
                        self._bdf.setStartdatetime(self._now)



                  ch_names_in_port_data = True
                  for ch_name in self._ch_names:
                    if not(ch_name in port.data):
                      ch_names_in_port_data = False
                  if ch_names_in_port_data:
                    #print(port.data)
                    self._bdf.writeSamples(port.data[self._ch_names].to_numpy().T)

    #                    if isinstance(port.data, pd.DataFrame):
    #                        port.data.index.freq = None
    #                    self._store.append(key, port.data, min_itemsize=self.min_itemsize)
    #                if port.meta is not None and port.meta:
    # Note: not none and not an empty dict, because this operation
    #       overwrites previous metadata and an empty dict would
    #       just remove any previous change
    #                    node = self._store.get_node(key)
    #                    if node:
    #                        self._store.get_node(key)._v_attrs["meta"] = port.meta

    def terminate(self):
        try:
            self._bdf.close()
#            print('saved bdf closed')
        except Exception:
            # Just in case
            pass
