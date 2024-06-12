"""Simple example nodes"""

from timeflux.core.node import Node


class Rereferencing(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(
        self,
        ref_channels,
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        self._ref_channels = ref_channels
        
        import numpy as np

    def update(self):
        import numpy as np

        if not self.i.ready():
            return

        self.o.meta = self.i.meta
        self.o.data = self.i.data
        if self.i.data is not None:
            data_columns = list(self.o.data.columns.values.tolist())
            for ref_channel in self._ref_channels:
                data_columns.remove(ref_channel)
#            self.o.data = self.o.data[data_columns]

            for_reref = self.i.data[self._ref_channels]
#            reref = None
#            for (colname, colval) in for_reref.items():
#                if reref is None:
#                    reref = colval / len(self._ref_channels)
#                else:
#                    reref = reref + colval / len(self._ref_channels)
            for (colname, colval) in self.o.data.items():
              for (colname_for_reref, colval_for_reref) in for_reref.items():
                self.o.data = self.o.data.assign(colname=self.o.data[colname] - self.o.data[colname_for_reref]) 

#                self.o.data[colname] = self.o.data[colname] - reref
            self.o.data = self.o.data[data_columns]
            #print(self.i.data, self.o.data)

