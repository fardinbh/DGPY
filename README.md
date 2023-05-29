# DGPY Library

DGPY is a Python library that provides a computational graph framework for building and executing deep learning models. It provides a set of classes to create and connect nodes representing mathematical operations and activations. The library supports forward propagation and backpropagation for gradient computation.

### To use the DGPY library, follow these steps:

Install Python: Make sure you have Python installed on your system. You can download Python from the official website: Python.org.
Install NumPy: DGPY depends on the NumPy library for array operations. Install NumPy by running the following command in your terminal or command prompt:
  
```pip install numpy```
Download DGPY library: Copy the code for the DGPY library provided in the question and save it to a Python file, for example, dgpy.py.

Start using DGPY: You can now import the DGPY library in your Python code and start using it. For example:

## Usage

The DGPY library provides several classes to create and connect nodes in a computation graph. Here's an overview of the available classes:

    * Graph: Represents a computation graph and provides methods for managing nodes, calculating values, and performing backpropagation.

    Node: Base class for all nodes in the graph. It defines the basic functionality and interface for nodes.

    ParameterNode: A type of node that holds a parameter value.

    TrainableNode: A type of parameter node that supports setting a learning rate.

    Mathematical operation nodes: AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode.

    Activation nodes: ReLUActivationNode, SigmoidActivationNode, TanhActivationNode.

    Other nodes: ExponentialNode, LogarithmNode, MatrixMultiplicationNode, Convolution2DNode, MaxPooling2DNode, FlattenNode, DropoutNode.

To use the DGPY library, you can create an instance of the Graph class and add nodes to it. Then, you can connect the nodes together to define the computation graph. Finally, you can call the calculate_topological_order method to perform the forward propagation and obtain the calculated values.
