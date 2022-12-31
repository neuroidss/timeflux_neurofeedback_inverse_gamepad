"""Simple example nodes"""

from timeflux.core.node import Node


class GephiStreamer(Node):

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

        # Basic import
        from gephistreamer import graph
        from gephistreamer import streamer

        # Create a Streamer
        # adapt if needed : streamer.GephiREST(hostname="localhost", port=8080, workspace="workspace0")
        self._stream = streamer.Streamer(streamer.GephiREST())
        self._nodes = {}
        self._edges = {}
        self._update = False

    def update(self):

        from gephistreamer import graph
        from gephistreamer import streamer

        # Create a node with a custom_property
        #        self._node_a = graph.Node("A",custom_property=1)

        # Create a node and then add the custom_property
        #        self._node_b = graph.Node("B")
        #        self._node_b.property['custom_property']=2

        # Add the node to the stream
        # you can also do it one by one or via a list
        # l = [node_a,node_b]
        # stream.add_node(*l)
        #        self._stream.add_node(self._node_a,self._node_b)

        # Create edge
        # You can also use the id of the node :  graph.Edge("A","B",custom_property="hello")
        #        self._edge_ab = graph.Edge(self._node_a,self._node_b,custom_property="hello")
        #        self._stream.add_edge(self._edge_ab)

        if self.ports is not None:
            for name, port in self.ports.items():
                #                if not name.startswith("i"):
                #                    continue
                #                key = "/" + name[2:].replace("_", "/")
                if port.data is not None:
                    #                print('port.data.iteritems():',port.data.iteritems())
                    for (colname, colval) in port.data.items():
                        nodes_names = colname.split("__")
                        self._nodes[nodes_names[0]] = graph.Node(nodes_names[0])
                        self._nodes[nodes_names[1]] = graph.Node(nodes_names[1])
                        #                  if self._nodes[nodes_names[0]] and self._nodes[nodes_names[1]]:
                        #                  if self._update:
                        #                     self._stream.change_node(self._nodes[nodes_names[0]],self._nodes[nodes_names[1]])
                        #                  else:
                        #                    self._stream.add_node(self._nodes[nodes_names[0]],self._nodes[nodes_names[1]])
                        val = colval.values[len(colval.values) - 1]
                        self._edges[colname] = graph.Edge(
                            self._nodes[nodes_names[0]],
                            self._nodes[nodes_names[1]],
                            red=val,
                            green=val,
                            blue=val,
                            directed=False,
                            weight=val,
                        )
                    #                  self._edges[colname] = graph.Edge(self._nodes[nodes_names[0]],self._nodes[nodes_names[1]],coherence=val)
                    #                  if self._edges[colname]:
                    #                  if self._update:
                    #                    self._stream.change_edge(self._edges[colname])
                    #                  else:
                    #                    self._stream.add_edge(self._edges[colname])
                    nodes = self._nodes.values()
                    edges = self._edges.values()
                    if self._update:
                        self._stream.change_node(*nodes)
                        self._stream.change_edge(*edges)
                    else:
                        self._stream.add_node(*nodes)
                        self._stream.add_edge(*edges)
                    self._update = True
