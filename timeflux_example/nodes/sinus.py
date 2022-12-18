"""timeflux_example.nodes.sinus: generate sinusoidal signal"""

import numpy as np
from timeflux.core.node import Node
from timeflux.helpers.clock import time_to_float, float_to_time, now
from timeflux.core.registry import Registry


class Sinus(Node):
    """Return a sinusoidal signal sampled to registry rate.

    This node generates a sinusoidal signal of chosen frequency and amplitude.
    Note that at each update, the node generate one row, so its sampling rate
    equals the graph parsing rate (given by the Registry).

    Attributes:
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/sinus.yaml
           :language: yaml

    .. deprecated::
        Use :func:`timeflux_example.nodes.signal.Sine` instead.

    """

    def __init__(self, amplitude=1, rate=1, name="sinus"):
        self._amplitude = amplitude
        self._rate = rate
        self._name = name
        self._start = None

    def update(self):
        timestamp = now()
        float = time_to_float(timestamp)
        if self._start is None:
            self._start = float

        values = [
            self._amplitude * np.sin(2 * np.pi * self._rate * (float - self._start))
        ]
        self.o.set(values, names=[self._name])
        self.o.meta = {"rate": Registry.rate}
