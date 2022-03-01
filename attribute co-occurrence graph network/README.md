# Node2vec

#### Installation

- Clone this repo.
- enter the directory where you clone it, and run the following code
    ```bash
    pip install -r requirements.txt
    cd src
    ```

#### Options
    python main.py --help


- --input, the input file of a network;
- --graph-format, the format of input graph, adjlist or edgelist;
- --output, the output file of representation (GCN doesn't need it);
- --representation-size, the number of latent dimensions to learn for each node; the default is 128
- --method, the NE model to learn, including deepwalk, line, node2vec, grarep, tadw, gcn, lap, gf, hope and sdne;
- --directed, treat the graph as directed; this is an action;
- --weighted, treat the graph as weighted; this is an action;
- --number-walks, the number of random walks to start at each node; the default is 10;
- --walk-length, the length of random walk started at each node; the default is 80;
- --workers, the number of parallel processes; the default is 8;
- --window-size, the window size of skip-gram model; the default is 10;
- --q, only for node2vec; the default is 1.0;
- --p, only for node2vec; the default is 1.0;

#### Example

To run "node2vec" on our dataset:

    python main.py --input attribute_graph.txt --graph-format edgelist --output node2vec_embedding.txt --weighted

#### Input
The supported input format is an edgelist or an adjlist:

    edgelist: node1 node2 <weight_float, optional>
    adjlist: node n1 n2 n3 ... nk
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

If the model needs additional features, the supported feature input format is as follow (**feature_i** should be a float number):

    node feature_1 feature_2 ... feature_n


#### Output
The output file has *n+1* lines for a graph with *n* nodes. 
The first line has the following format:

    num_of_nodes  dim_of_representation

The next *n* lines are as follows:
    
    node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation.
