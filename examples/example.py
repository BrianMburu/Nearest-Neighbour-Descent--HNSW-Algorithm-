import numpy as np
from hnsw import HNSW

# Initialize the HNSW class

hnsw = HNSW()

# Build an index using known embeddings

known_embeddings = [...] # A list of NumPy arrays
hnsw.initialize(known_embeddings)
known_labels = [...] # A list of labels corresponding to the embeddings

# Query the index for the closest neighbor to a test embedding

test_embedding = [...] # A NumPy array
closest_label, closest_distance = hnsw.get_closest_neighbour(test_embedding, known_labels)

# Test the performance of the HNSW algorithm on a test dataset

test_embeddings = [...] # A list of NumPy arrays
test_labels = [...]    # A list of labels corresponding to the embeddings
accuracy, mis_labeled, total_time = hnsw.test_performance(test_embeddings, test_labels, known_labels)