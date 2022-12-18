"""timeflux_example.nodes.signal: generate signals"""

import time
import numpy as np
from timeflux.core.node import Node


class Sine(Node):
    """Generate a sinusoidal signal.

    Attributes:
        o (Port): Default output, provides DataFrame.

    Args:
        frequency (float): cycles per second. Default: ``1``.
        resolution (int): points per second. Default: ``200``.
        amplitude (float): signal amplitude. Default: ``1``.
        name (string): signal name. Default: ``sine``.

    Example:
        .. literalinclude:: /../examples/sine.yaml
           :language: yaml
    """

    def __init__(self, frequency=1, resolution=200, amplitude=1, name="sine"):
        self._frequency = frequency
        self._resolution = int(resolution)
        self._amplitude = amplitude
        self._name = name
        self._radian = 0
        self._now = time.time()

    def update(self):
        now = time.time()
        elapsed = now - self._now
        points = int(elapsed * self._resolution) + 1
        cycles = self._frequency * elapsed
        values = np.linspace(self._radian, np.pi * 2 * cycles + self._radian, points)
        signal = np.sin(values[:-1]) * self._amplitude
        timestamps = np.linspace(
            int(self._now * 1e6), int(now * 1e6), points, False, dtype="datetime64[us]"
        )[1:]
        self._now = now
        self._radian = values[-1]
        self.o.set(signal, timestamps, names=[self._name])
        self.o.meta = {"rate": self._frequency}
