

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import gudhi as gd


# Helper function to get the hidden states of a specific layer for a given input sequence
def get_hidden_states(tokenizer, model, layer, input_sequence):
    model.config.output_hidden_states = True
    encoded_input = tokenizer([input_sequence], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**encoded_input)
    hidden_states = model_output.hidden_states
    specific_hidden_states = hidden_states[layer][0]
    return specific_hidden_states

# Helper function to compute the Euclidean distance matrix
def compute_euclidean_distance_matrix_scipy(hidden_states):
    euclidean_distances = pdist(hidden_states.numpy(), metric=euclidean)
    euclidean_distance_matrix = squareform(euclidean_distances)
    return euclidean_distance_matrix


# Helper function to compute the persistent homology of a given distance matrix
def compute_persistent_homology(distance_matrix, max_dimension=3):
    max_edge_length = np.max(distance_matrix)
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    persistence = st.persistence()
    return st, persistence

# Helper function to compute the Wasserstein distances between all pairs of persistence diagrams
def compute_wasserstein_distances(persistence_diagrams, dimension):
    n_diagrams = len(persistence_diagrams)
    distances = np.zeros((n_diagrams, n_diagrams))
    filtered_diagrams = [[point for point in diagram if point[0] == dimension] for diagram in persistence_diagrams]
    for i in range(n_diagrams):
        for j in range(i+1, n_diagrams):
            X = np.array([p[1][1] - p[1][0] for p in filtered_diagrams[i] if p[1][1] != float('inf')])
            Y = np.array([p[1][1] - p[1][0] for p in filtered_diagrams[j] if p[1][1] != float('inf')])
            distance = wasserstein_distance(X, Y)
            distances[i][j] = distance
            distances[j][i] = distance
    return distances

