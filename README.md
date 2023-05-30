# DYGPY Library

DYGPY is a Python library that provides a computational dynamic graph framework for building and executing deep learning models. It provides a set of classes to create and connect nodes representing mathematical operations and activations. The library supports forward propagation and backpropagation for gradient computation.

### To use the DYGPY library, follow these steps:

`1.` Install Python: Make sure you have Python installed on your system. You can download Python from the official website: Python.org.<br>
`2.` Install NumPy: DYGPY depends on the NumPy library for array operations. Install NumPy by running the following command in your terminal or command prompt:<br> 
```pip install numpy```<br>
`3.` Download DYGPY library<br>

`4.` Start using DYGPY: You can now import the DYGPY library in your Python code and start using it.<br>
## Usage

The DYGPY library provides several classes to create and connect nodes in a computation graph. Here's an overview of the available classes:

1. **Graph Class**
The `Graph` class represents a computational graph. It provides methods for managing nodes, connecting them, calculating values, updating gradients, and finding the topological order of the nodes in the graph. The main methods of the `Graph` class are:

- `add_node(node)`: Adds a node to the graph.
- `connect(source, target)`: Connects two nodes in the graph, making the target node a child of the source node.
- `calculate()`: Calculates the values of all nodes in the graph by performing a forward pass.
- `_calculate_node(node)`: Helper method for calculating the value of a single node.
- `update_gradients()`: Updates the gradients of all nodes in the graph by performing a backward pass.
- `clear_gradients()`: Clears the gradients of all nodes in the graph.
- `find_topological_order()`: Finds the topological order of the nodes in the graph, which is the order in which the nodes should be calculated.
- `calculate_topological_order()`: Calculates the values of all nodes in the graph in the topological order.
- `get_parameters()`: Returns a list of parameter nodes in the graph.
- `get_trainable_parameters()`: Returns a list of trainable nodes in the graph.
- `set_learning_rate(learning_rate)`: Sets the learning rate for all trainable nodes in the graph.

2. **Node Class**
The `Node` class represents a node in the computational graph. It serves as the base class for different types of nodes. Each node has a value, a list of children nodes, and a gradient. The main methods of the `Node` class are:

- `add_child(child)`: Adds a child node to the current node.
- `update_value(value)`: Updates the value of the node and propagates the change to its children.
- `propagate()`: Propagates the value change to the children nodes.
- `calculate()`: Calculates the value of the node. This method is overridden by specific node types.
- `update_gradients()`: Updates the gradients of the node and its children.
- `clear_gradients()`: Clears the gradients of the node.

3. **GraphWithDifferentiation Class**
The `GraphWithDifferentiation` class is a subclass of `Graph` that adds support for automatic differentiation. It overrides the `calculate()` and `calculate_topological_order()` methods to also update gradients after the forward pass.

4. **Specific Node Classes**
The library provides various node classes representing different operations or activations that can be performed on the values in the graph. These include `ParameterNode`, `TrainableNode`, `AdditionNode`, `SubtractionNode`, `MultiplicationNode`, `DivisionNode`, `PowerNode`, `ExponentialNode`, `LogarithmNode`, `ReLUActivationNode`, `SigmoidActivationNode`, `TanhActivationNode`, `MatrixMultiplicationNode`, `Convolution2DNode`, `MaxPooling2DNode`, `FlattenNode`, `DropoutNode`, and more. Each specific node class inherits from the `Node` class and overrides the `calculate()` method to define its specific calculation.

5. **Main Function**
The `main()` function serves as an example of how to use the library. It demonstrates the creation of a graph, the instantiation of various nodes, connecting them, performing calculations.<br>

To use the DYGPY library, you can create an instance of the Graph class and add nodes to it. Then, you can connect the nodes together to define the computation graph. Finally, you can call the calculate_topological_order method to perform the forward propagation and obtain the calculated values.<br>


### Dependencies

The DYGPY library has the following dependencies:<br>

`numpy`

Make sure to install the dependencies before using the library.<br>

### License

The DYGPY library is released under the MIT License. You can find the full license text in the LICENSE file.<br>

### Contact

For any questions or support regarding the DYGPY library, please contact the author at fardinbhi@gmail.com.
