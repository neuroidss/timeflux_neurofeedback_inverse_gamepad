"""Simple example nodes"""

from timeflux.core.node import Node


class UInput(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self):
        """
        Args:
            value (int): The value to add to each cell.
        """
#        self._value = value
        import uinput
#        print('uinput init')
#        import time
        import numpy as np
        self._events = (
            uinput.BTN_A,
            uinput.BTN_B,
            uinput.BTN_X,
            uinput.BTN_Y,
            uinput.BTN_TL,
            uinput.BTN_TR,
            uinput.BTN_TL2,
            uinput.BTN_TR2,
            uinput.BTN_DPAD_UP,
            uinput.BTN_DPAD_DOWN,
            uinput.BTN_DPAD_LEFT,
            uinput.BTN_DPAD_RIGHT,
            uinput.BTN_SELECT,
            uinput.BTN_START,
            uinput.BTN_MODE,
            uinput.BTN_THUMBL,
            uinput.BTN_THUMBR,
            uinput.ABS_X + (0, 0x8000, 0, 0),
            uinput.ABS_Y + (0, 0x8000, 0, 0),
            uinput.ABS_Z + (0, 0x8000, 0, 0),
            uinput.ABS_RX + (0, 0x8000, 0, 0),
            uinput.ABS_RY + (0, 0x8000, 0, 0),
            uinput.ABS_RZ + (0, 0x8000, 0, 0),
            uinput.REL_X + (0, 0x8000, 0, 0),
            uinput.REL_Y + (0, 0x8000, 0, 0),
            uinput.REL_Z + (0, 0x8000, 0, 0),
            uinput.REL_RX + (0, 0x8000, 0, 0),
            uinput.REL_RY + (0, 0x8000, 0, 0),
            uinput.REL_RZ + (0, 0x8000, 0, 0),
#        uinput.ABS_X + (0, 255, 0, 0),
#        uinput.ABS_Y + (0, 255, 0, 0),
#        uinput.ABS_Z + (0, 255, 0, 0),
#        uinput.ABS_RX + (0, 255, 0, 0),
#        uinput.ABS_RY + (0, 255, 0, 0),
#        uinput.ABS_RZ + (0, 255, 0, 0),
#        uinput.REL_X + (0, 255, 0, 0),
#        uinput.REL_Y + (0, 255, 0, 0),
#        uinput.REL_Z + (0, 255, 0, 0),
#        uinput.REL_RX + (0, 255, 0, 0),
#        uinput.REL_RY + (0, 255, 0, 0),
#        uinput.REL_RZ + (0, 255, 0, 0),
        )
        self._uinput_device = uinput.Device(self._events)
#        print('uinput init ok')
	
    def update(self):
        import uinput
        import numpy as np
#        print('uinput update, self.ports:',self.ports)
        if self.ports is not None:
#          bdf.writeSamples(bufs_hstack_cut)
            for name, port in self.ports.items():
#                if not name.startswith("i"):
#                    continue
#                key = "/" + name[2:].replace("_", "/")
              if port.data is not None:
#                print('port.data.iteritems():',port.data.iteritems())
                for (colname,colval) in port.data.items():
#                  print('colname, colval.values:',colname, colval.values)
#                  if np.max(colval.values)-np.min(colval.values) == 0:
#                    val = np.nan
#                  else:
#                    val = (colval.values[len(colval.values)-1]-np.min(colval.values))/(np.max(colval.values)-np.min(colval.values))
                  val = colval.values[len(colval.values)-1]
#                  print('val:',val)
                  if not np.isnan(val):
#                  for idx0, gamepad_data in enumerate(joy_gamepad_scores_data_combined):
#                   if not np.isnan(scores_shifts_baselined_combined[idx0]):
                    if colname == 'XR':
                      self._uinput_device.emit(uinput.ABS_RX, round(0x8000 * val))
                    if colname == 'YR':
                      self._uinput_device.emit(uinput.ABS_RY, round(0x8000 * val))
                    if colname == 'ZR':
                      self._uinput_device.emit(uinput.ABS_RZ, round(0x8000 * val))
                    if colname == 'X':
                      self._uinput_device.emit(uinput.ABS_X, round(0x8000 * val))
                    if colname == 'Y':
                      self._uinput_device.emit(uinput.ABS_Y, round(0x8000 * val))
                    if colname == 'Z':
                      self._uinput_device.emit(uinput.ABS_Z, round(0x8000 * val))
                    if (colname.find('B') == 0) and (int(colname[1:]) >= 1) and (int(colname[1:]) <= 17):
                      if round(val) > 0:
                        self._uinput_device.emit((uinput.BTN_GAMEPAD[0],uinput.BTN_GAMEPAD[1]+(int(colname[1:])-1)), 1)
                      else:
                        self._uinput_device.emit((uinput.BTN_GAMEPAD[0],uinput.BTN_GAMEPAD[1]+(int(colname[1:])-1)), 0)                
#                  self._bdf.writeSamples(port.data[self._ch_names].to_numpy().T)

class VJoy(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self):
        """
        Args:
            value (int): The value to add to each cell.
        """
#        self._value = value
        import pyvjoy
        import numpy as np
        self._vjoy_device = pyvjoy.VJoyDevice(1)
#        self._vjoy.data.lButtons = 0
#        self._vjoy.data.wAxisXRot = round(0x8000 / 2)
#        self._vjoy.data.wAxisYRot = round(0x8000 / 2)
#        self._vjoy.data.wAxisZRot = round(0x8000 / 2)
#        self._vjoy.data.wAxisX = round(0x8000 / 2)
#        self._vjoy.data.wAxisY = round(0x8000 / 2)
#        self._vjoy.data.wAxisZ = round(0x8000 / 2)
#        self._vjoy.update()
	
    def update(self):
        import pyvjoy
        import numpy as np
        if self.ports is not None:
#          bdf.writeSamples(bufs_hstack_cut)
            for name, port in self.ports.items():
#                if not name.startswith("i"):
#                    continue
#                key = "/" + name[2:].replace("_", "/")
              if port.data is not None:
                for (colname,colval) in port.data.items():
#                  print(colname, colval.values)
#                  if np.max(colval.values)-np.min(colval.values) == 0:
#                    val = np.nan
#                  else:
#                    val = (colval.values[len(colval.values)-1]-np.min(colval.values))/(np.max(colval.values)-np.min(colval.values))
                  val = colval.values[len(colval.values)-1]
                  if not np.isnan(val):
                    if colname == 'XR':
                      self._vjoy_device.data.wAxisXRot = round(0x8000 * val)
                    if colname == 'YR':
                      self._vjoy_device.data.wAxisYRot = round(0x8000 * val)
                    if colname == 'ZR':
                      self._vjoy_device.data.wAxisZRot = round(0x8000 * val)
                    if colname == 'X':
                      self._vjoy_device.data.wAxisX = round(0x8000 * val)
                    if colname == 'Y':
                      self._vjoy_device.data.wAxisY = round(0x8000 * val)
                    if colname == 'Z':
                      self._vjoy_device.data.wAxisZ = round(0x8000 * val)
                    if (colname.find('B') == 0) and (int(colname[1:]) >= 1) and (int(colname[1:]) <= 8):
                      if round(val) > 0:
                        self._vjoy_device.data.lButtons |= (1<<(int(colname[1:])-1))
                      else:
                        self._vjoy_device.data.lButtons &= ~(1<<(int(colname[1:])-1))
                  self._vjoy_device.update()
    

