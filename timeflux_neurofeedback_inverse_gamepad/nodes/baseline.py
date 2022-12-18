"""Simple example nodes"""

from timeflux.core.node import Node


class Baseline(Node):

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
        import numpy as np
	
    def update(self):
        import numpy as np
        
        if not self.i.ready():
            return

        self.o.meta = self.i.meta
        self.o.data = self.i.data.tail(1)
        if self.i.data is not None:
                for (colname,colval) in self.i.data.items():
#                  print('colname, colval.values:',colname, colval.values)
                  if np.max(colval.values)-np.min(colval.values) == 0:
                    val = np.nan
                  else:
                    val = (colval.values[len(colval.values)-1]-np.min(colval.values))/(np.max(colval.values)-np.min(colval.values))
#                  print('self.o.data:',self.o.data)
#                  print('self.o.data.iloc[0].at[colname]:',self.o.data.iloc[0].at[colname])
                    self.o.data.iloc[0].at[colname] = val

