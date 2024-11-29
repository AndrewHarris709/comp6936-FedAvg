# Federated

This package contains client and server classes.

## FederatedClient

Each client has the following attributes:

- **name**

    Name of the client

- **model**

    The linear model on the client

- **generator**

    Data generator used for generating training data

- **failure_rate**

    Rate which failure or outage will happen on the client

- **rng**
    
    Used for random generation

- **initial**

    Initial data points at the start of process

- **data_X**

    Data points on the client

- **data_Y**

    Data labels on the client

Each client will have the following methods:

- **add_data**

    This method gets a new data point and label from the generator and adds it to data_X and data_Y.

- **train**

    This method starts the training process for the client and returns model parameters after training.

- **reset**

    This method resets the client and set attibutes to the initial values.

## FederatedServer

The server has the following attributes:

- **C**

    Clients participation ratio

- **clients**

    List of client ids (ids are assigned when creating sockets using socket.io)

- **m**

    Number of selected clients for training

- **model**

    Centeralized model on the server

- **clientWs**

    List of clients' model parameters

- **clientWRecord**

    Dictionary with client ids as keys and their model parameters as values

- **initialWeights**

    Initial model parameters for the start

- **max_iter**

    Maximum number of iteration for model training on the clients

The server has the following methods:

- **select_clients**

    This method randomly selects and returns clients based on participation ratio for training.

- **update_weights**

    This method receives clients' model parameters after training as input, aggregates them based on the number data points each client was trained on and updates the centeralized model. 

- **update_centralized_model**

    This methods receives model parameters as input and updates the server model parameters accordingly.


- **reset**

    This method resets the server and set attibutes to the initial values.