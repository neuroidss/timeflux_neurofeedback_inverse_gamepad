"""Simple example nodes"""

from timeflux.core.node import Node


class GradioStreamer(Node):

    """Adds ``value`` to each cell of the input.

    This is one of the simplest possible nodes.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../examples/test.yaml
           :language: yaml
    """

    def __init__(self,
        hostname = '127.0.0.1',
        port = '7860',
        fn_index = 2,
        attention_type = "coherence",
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value

        # Basic import
        from gradio_client import Client

        # Create a Streamer
        # adapt if needed : streamer.GephiREST(hostname="localhost", port=8080, workspace="workspace0")
        self._hostname = hostname
        self._port = port
        self._client = Client("http://"+self._hostname+":"+self._port+"/")
        self._fn_index = fn_index
        self._attention_type = attention_type

    def update(self):
    
        from gradio_client import Client
        import json
        from json import JSONEncoder

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
                    port_data_values_as_list = port.data.values[-1:].tolist()
                    encodedData = json.dumps(port_data_values_as_list[0])
                    result = self._client.predict(
				self._attention_type,	# str  in 'Attention Type' Radio component
				encodedData,	# str  in 'Coherence JSON' Textbox component
				fn_index=self._fn_index
                                )
                    #print(result)

