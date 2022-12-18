"""Illustrates dynamic inputs and outputs."""

import random
from timeflux.core.node import Node


class Outputs(Node):

    """Randomly generate dynamic outputs.

    At each update, this node generates a random number of outputs and sets the default output
    to the number it has created.

    Attributes:
        o (Port): Default output, provides DataFrame.
        o_* (Port): Dynamic outputs.

    Args:
        seed (int): The random number generator seed.
        prefix (string): The prefix to add to each dynamic output.

    Example:
        .. literalinclude:: /../examples/dynamic_prefixed.yaml
           :language: yaml
    """

    def __init__(self, prefix=None, seed=None):
        random.seed(seed)
        self.prefix = "" if prefix is None else prefix + "_"

    def update(self):
        # Lazily create new ports
        for i in range(random.randint(0, 10)):
            getattr(self, "o_" + self.prefix + str(i))
        # Count
        outputs = len(list(self.iterate("o_*")))
        # Set default output
        self.o.set([[outputs]], names=["outputs"])


class Inputs(Node):

    """Count the dynamic outputs.

    At each update, this node loops over all dynamic inputs and sets the default output
    to the number it has found.

    Attributes:
        i_* (Port): Dynamic inputs.
        o (Port): Default output, provides DataFrame.

    Args:
        prefix (string): The prefix to add to match dynamic inputs.

    Example:
        .. literalinclude:: /../examples/dynamic.yaml
           :language: yaml
    """

    def __init__(self, prefix=None):
        self.prefix = "" if prefix is None else prefix + "_"

    def update(self):
        # Count
        inputs = len(list(self.iterate("i_" + self.prefix + "*")))
        # Set default output
        self.o.set([[inputs]], names=["inputs"])
