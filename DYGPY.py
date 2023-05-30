import numpy as np


class Graph:
    def __init__(self):
        self.nodes = []
    
    def add_node(self, node):
        self.nodes.append(node)
    
    def connect(self, source, target):
        source.add_child(target)
    
    def calculate(self):
        for node in self.nodes:
            self._calculate_node(node)
    
    def _calculate_node(self, node):
        node.calculate()
    
    def update_gradients(self):
        for node in self.nodes:
            node.update_gradients()
    
    def clear_gradients(self):
        for node in self.nodes:
            node.clear_gradients()
    
    def find_topological_order(self):
        order = []
        visited = set()
        
        def dfs(node):
            visited.add(node)
            
            for child in node.children:
                if child not in visited:
                    dfs(child)
            
            order.append(node)
        
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        return order

    def calculate_topological_order(self):
        order = self.find_topological_order()
        
        for node in order:
            self._calculate_node(node)

    def get_parameters(self):
        parameters = []
        
        for node in self.nodes:
            if isinstance(node, ParameterNode):
                parameters.append(node)
        
        return parameters

    def get_trainable_parameters(self):
        trainable_parameters = []
        
        for node in self.nodes:
            if isinstance(node, TrainableNode):
                trainable_parameters.append(node)
        
        return trainable_parameters

    def set_learning_rate(self, learning_rate):
        for node in self.get_trainable_parameters():
            node.set_learning_rate(learning_rate)


class Node:
    def __init__(self, value):
        self.value = np.array(value)
        self.children = []
        self.gradient = None
    
    def add_child(self, child):
        self.children.append(child)
    
    def update_value(self, value):
        self.value = np.array(value)
        self.propagate()
    
    def propagate(self):
        for child in self.children:
            child.calculate()
    
    def calculate(self):
        pass
    
    def update_gradients(self):
        if self.gradient is not None:
            for child in self.children:
                child.gradient = self.gradient
    
    def clear_gradients(self):
        self.gradient = None


class GraphWithDifferentiation(Graph):
    def calculate(self):
        super().calculate()
        self.update_gradients()
    
    def calculate_topological_order(self):
        order = self.find_topological_order()
        
        for node in order:
            self._calculate_node(node)
        
        self.update_gradients()


class ParameterNode(Node):
    def __init__(self, value):
        super().__init__(value)
    
    def set_value(self, value):
        self.update_value(value)


class TrainableNode(ParameterNode):
    def __init__(self, value):
        super().__init__(value)
        self.learning_rate = 0.01
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class AdditionNode(Node):
    def __init__(self, *inputs):
        super().__init__(np.array(0.0))
        self.inputs = inputs
    
    def calculate(self):
        self.value = np.sum([input_node.value for input_node in self.inputs], axis=0)


class SubtractionNode(Node):
    def __init__(self, input_a, input_b):
        super().__init__(np.array(0.0))
        self.input_a = input_a
        self.input_b = input_b
    
    def calculate(self):
        self.value = self.input_a.value - self.input_b.value


class MultiplicationNode(Node):
    def __init__(self, *inputs):
        super().__init__(np.array(1.0))
        self.inputs = inputs
    
    def calculate(self):
        self.value = np.prod([input_node.value for input_node in self.inputs], axis=0)


class DivisionNode(Node):
    def __init__(self, input_a, input_b):
        super().__init__(np.array(1.0))
        self.input_a = input_a
        self.input_b = input_b
    
    def calculate(self):
        self.value = self.input_a.value / self.input_b.value


class PowerNode(Node):
    def __init__(self, input_node, exponent):
        super().__init__(np.array(1.0))
        self.input_node = input_node
        self.exponent = exponent
    
    def calculate(self):
        self.value = np.power(self.input_node.value, self.exponent)


class ExponentialNode(Node):
    def __init__(self, input_node):
        super().__init__(np.array(0.0))
        self.input_node = input_node
    
    def calculate(self):
        self.value = np.exp(self.input_node.value)


class LogarithmNode(Node):
    def __init__(self, input_node):
        super().__init__(np.array(0.0))
        self.input_node = input_node
    
    def calculate(self):
        self.value = np.log(self.input_node.value)


class ReLUActivationNode(Node):
    def calculate(self):
        self.value = np.maximum(0, self.value)


class SigmoidActivationNode(Node):
    def calculate(self):
        self.value = 1 / (1 + np.exp(-self.value))


class TanhActivationNode(Node):
    def calculate(self):
        self.value = np.tanh(self.value)


class MatrixMultiplicationNode(Node):
    def __init__(self, *inputs):
        super().__init__(np.array([[0.0]]))
        self.inputs = inputs
    
    def calculate(self):
        if len(self.inputs) != 2:
            raise ValueError("MatrixMultiplicationNode requires exactly two input nodes.")
        
        matrix_a = self.inputs[0].value
        matrix_b = self.inputs[1].value
        
        self.value = np.matmul(matrix_a, matrix_b)


class Convolution2DNode(Node):
    def __init__(self, input_node, filter_node, stride=1, padding=0):
        super().__init__(np.zeros((1, 1, 1, 1)))
        self.input_node = input_node
        self.filter_node = filter_node
        self.stride = stride
        self.padding = padding
    
    def calculate(self):
        input_data = self.input_node.value
        filter_data = self.filter_node.value
        batch_size, input_channels, input_height, input_width = input_data.shape
        filter_channels, _, filter_height, filter_width = filter_data.shape
        
        output_height = ((input_height - filter_height + 2 * self.padding) // self.stride) + 1
        output_width = ((input_width - filter_width + 2 * self.padding) // self.stride) + 1
        output_shape = (batch_size, filter_channels, output_height, output_width)
        
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        output_data = np.zeros(output_shape)
        
        for b in range(batch_size):
            for c in range(filter_channels):
                for i in range(0, input_height, self.stride):
                    for j in range(0, input_width, self.stride):
                        receptive_field = padded_input[b, :, i:i+filter_height, j:j+filter_width]
                        output_data[b, c, i//self.stride, j//self.stride] = np.sum(receptive_field * filter_data[c])
        
        self.value = output_data


class MaxPooling2DNode(Node):
    def __init__(self, input_node, pool_size=2, stride=None, padding=0):
        super().__init__(np.zeros((1, 1, 1, 1)))
        self.input_node = input_node
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.padding = padding
    
    def calculate(self):
        input_data = self.input_node.value
        batch_size, input_channels, input_height, input_width = input_data.shape
        
        output_height = ((input_height - self.pool_size + 2 * self.padding) // self.stride) + 1
        output_width = ((input_width - self.pool_size + 2 * self.padding) // self.stride) + 1
        output_shape = (batch_size, input_channels, output_height, output_width)
        
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        output_data = np.zeros(output_shape)
        
        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(0, input_height, self.stride):
                    for j in range(0, input_width, self.stride):
                        receptive_field = padded_input[b, c, i:i+self.pool_size, j:j+self.pool_size]
                        output_data[b, c, i//self.stride, j//self.stride] = np.max(receptive_field)
        
        self.value = output_data


class FlattenNode(Node):
    def calculate(self):
        self.value = np.reshape(self.value, (self.value.shape[0], -1))


class DropoutNode(Node):
    def __init__(self, input_node, dropout_rate=0.5, training_mode=True):
        super().__init__(np.zeros_like(input_node.value))
        self.input_node = input_node
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training_mode = training_mode
    
    def calculate(self):
        if self.training_mode:
            self.mask = np.random.binomial(1, self.dropout_rate, size=self.input_node.value.shape) / self.dropout_rate
            self.value = self.input_node.value * self.mask
        else:
            self.value = self.input_node.value
    
    def update_gradients(self):
        if self.gradient is not None:
            self.input_node.gradient = self.gradient * self.mask
    
    def clear_gradients(self):
        self.gradient = None
        self.input_node.clear_gradients()


def main():
    # Create a graph
    graph = GraphWithDifferentiation()
    
    # Create nodes
    node1 = ParameterNode(2)
    node2 = ParameterNode(3)
    node3 = AdditionNode(node1, node2)
    node4 = MultiplicationNode(node3, node1)
    node5 = ReLUActivationNode(node4)
    node6 = SigmoidActivationNode(node2)
    node7 = SubtractionNode(node1, node2)
    node8 = DivisionNode(node3, node1)
    node9 = PowerNode(node4, 2)
    node10 = ExponentialNode(node5)
    node11 = LogarithmNode(node6)
    node12 = TanhActivationNode(node7)
    
    # Add nodes to the graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_node(node5)
    graph.add_node(node6)
    graph.add_node(node7)
    graph.add_node(node8)
    graph.add_node(node9)
    graph.add_node(node10)
    graph.add_node(node11)
    graph.add_node(node12)
    
    # Connect nodes
    graph.connect(node1, node3)
    graph.connect(node2, node3)
    graph.connect(node3, node4)
    graph.connect(node4, node5)
    graph.connect(node2, node6)
    graph.connect(node1, node7)
    graph.connect(node3, node8)
    graph.connect(node4, node9)
    graph.connect(node5, node10)
    graph.connect(node6, node11)
    graph.connect(node7, node12)
    
    # Calculate the graph
    graph.calculate_topological_order()
    
    # Print the values of nodes
    print("Node 1:", node1.value)
    print("Node 2:", node2.value)
    print("Node 3:", node3.value)
    print("Node 4:", node4.value)
    print("Node 5:", node5.value)
    print("Node 6:", node6.value)
    print("Node 7:", node7.value)
    print("Node 8:", node8.value)
    print("Node 9:", node9.value)
    print("Node 10:", node10.value)
    print("Node 11:", node11.value)
    print("Node 12:", node12.value)
    
    # Perform backpropagation
    graph.clear_gradients()
    node5.gradient = np.ones_like(node5.value)
    graph.calculate_topological_order()
    
    # Print the gradients of nodes
    print("Node 1 gradient:", node1.gradient)
    print("Node 2 gradient:", node2.gradient)
    print("Node 3 gradient:", node3.gradient)
    print("Node 4 gradient:", node4.gradient)
    print("Node 5 gradient:", node5.gradient)
    print("Node 6 gradient:", node6.gradient)
    print("Node 7 gradient:", node7.gradient)
    print("Node 8 gradient:", node8.gradient)
    print("Node 9 gradient:", node9.gradient)
    print("Node 10 gradient:", node10.gradient)
    print("Node 11 gradient:", node11.gradient)
    print("Node 12 gradient:", node12.gradient)


if __name__ == '__main__':
    main()
