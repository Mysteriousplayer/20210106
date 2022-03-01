from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import node2vec
from graph import *
import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,#
                        help='Input graph file')
    parser.add_argument('--output',#
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',#
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    args = parser.parse_args()

    if not args.output:
        print("No output filename. Exit.")
        exit(1)

    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")

    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)

    model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                              num_paths=args.number_walks, dim=args.representation_size,
                              workers=args.workers, p=args.p, q=args.q, window=args.window_size)

    t2 = time.time()
    print(t2-t1)
    print("Saving embeddings...")
    model.save_embeddings(args.output)



if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
