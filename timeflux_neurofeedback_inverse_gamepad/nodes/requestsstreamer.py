"""Simple example nodes"""

from timeflux.core.node import Node


class RequestsStreamer(Node):

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
        fn_index = None,
        api_name = None,
        format_str = "sj",
        str_array = ["coherence"],
    ):
        """
        Args:
            value (int): The value to add to each cell.
        """
        #        self._value = value

        # Basic import
        import requests

        # Create a Streamer
        # adapt if needed : streamer.GephiREST(hostname="localhost", port=8080, workspace="workspace0")
        self._hostname = hostname
        self._port = port
        self._url = "http://"+self._hostname+":"+self._port+"/"
        self._fn_index = fn_index
        self._api_name = api_name
        self._format_str = format_str
        self._str_array = str_array
        
        url = self._url # or your URL
        opt = requests.get(url=f'{url}sdapi/v1/options')
        opt_json = opt.json()
#        print(opt_json)
        
    def update(self):
    
        import requests
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
            output_array=[]
            while (len(output_array)<len(self._format_str)) and ((self._format_str[len(output_array)]=='s')or(self._format_str[len(output_array)]=='f')):
                if (self._format_str[len(output_array)]=='s'):
                  output_array.append(self._str_array[len(output_array)])
                elif (self._format_str[len(output_array)]=='f'):
                  import os.path
                  if os.path.exists(self._str_array[len(output_array)]):
                    output_array.append(self._str_array[len(output_array)])
                  else:
                    return False
            for name, port in self.ports.items():
                #                if not name.startswith("i"):
                #                    continue
                #                key = "/" + name[2:].replace("_", "/")
                while (len(output_array)<len(self._format_str)) and ((self._format_str[len(output_array)]=='s')or(self._format_str[len(output_array)]=='f')):
                  if (self._format_str[len(output_array)]=='s'):
                    output_array.append(self._str_array[len(output_array)])
                  elif (self._format_str[len(output_array)]=='f'):
                    import os.path
                    if os.path.exists(self._str_array[len(output_array)]):
                      output_array.append(self._str_array[len(output_array)])
                    else:
                      return False
                if port.data is not None:
#                    print('port.data.values: ', port.data.values)
                    port_data_values_as_list = port.data.values[-1:].tolist()
                    encodedData = json.dumps(port_data_values_as_list[0])
                    if (len(output_array)<len(self._format_str)) and (self._format_str[len(output_array)]=='l'):
                      port_data_values_as_list_value_index = 0
                      while (len(output_array)<len(self._format_str)) and (self._format_str[len(output_array)]=='l'):
                        
                        encodedData = json.dumps(port_data_values_as_list[0][port_data_values_as_list_value_index])
                        port_data_values_as_list_value_index = port_data_values_as_list_value_index + 1
                        output_array.append(encodedData)
                    if (len(output_array)<len(self._format_str)) and (self._format_str[len(output_array)]=='j'):
                      encodedData = json.dumps(port_data_values_as_list[0])
                      output_array.append(encodedData)
#                    result = self._client.predict(
#				self._attention_type,	# str  in 'Attention Type' Radio component
#				encodedData,	# str  in 'Coherence JSON' Textbox component
#				fn_index=self._fn_index
#                                )
                    #print(result)
            while (len(output_array)<len(self._format_str)) and ((self._format_str[len(output_array)]=='s')or(self._format_str[len(output_array)]=='f')):
                if (self._format_str[len(output_array)]=='s'):
                  output_array.append(self._str_array[len(output_array)])
                elif (self._format_str[len(output_array)]=='f'):
                  import os.path
                  if os.path.exists(self._str_array[len(output_array)]):
                    output_array.append(self._str_array[len(output_array)])
                  else:
                    return False
            if len(output_array) == len(self._format_str):
#                print('len(output_array): ', len(output_array), 'self._format_str: ', self._format_str)
#                print('output_array: ', output_array)

                url = self._url

                option_payload = {
                    "attention_type": 'Coherence',
#                    "coherence_json": '{0,0,1,0,1,1}'
                    "coherence_json": json.dumps(port_data_values_as_list[0])
                }

                ##print(option_payload)
                response = requests.post(url=f'{url}sdapi/v1/options', json=option_payload)
                ##print(response)

#                result = self._client.predict(
#                self._client.submit(
#				*output_array,
#				fn_index=self._fn_index,
#				api_name=self._api_name,
#                                )
#                print('result: ', result)

