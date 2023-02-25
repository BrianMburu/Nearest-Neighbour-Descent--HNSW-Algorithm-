## HNSW Algorithm

This repository contains an implementation of the HNSW algorithm for approximate nearest neighbor search using the nmslib library.

### Overview

The HNSW algorithm is an approximate nearest neighbor search algorithm that constructs a hierarchical graph structure to index high-dimensional data points. This implementation of the HNSW algorithm is based on the NN-Descent algorithm and uses the cosine similarity metric.

### Requirements

- Python 3.x
- nmslib

### Installation

To install nmslib, run the following command:
`pip install nmslib`

### Usage

The HNSW class provides an implementation of the HNSW algorithm. The class has three main methods:

- **init**: Initializes an instance of the HNSW class.
- initialize: Builds an index for the HNSW algorithm based on known embeddings.
- get_closest_neighbour: Finds the closest neighbor to a query embedding in the HNSW index.
- test_performance: Tests the performance of the HNSW algorithm on a test dataset.

### Example

```python
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
```

### Time Complexity
- **init**: O(1)
- initialize: O(n log n), where n is the number of known embeddings.
- get_closest_neighbour: O(log n), where n is the number of known embeddings.
- test_performance: O(m log n), where m is the number of test embeddings and n is the number of known embeddings.

It's worth noting that these time complexities are based on the performance of the underlying nmslib library and assume a cosine similarity distance metric. Other distance metrics may have different time complexities.

### References

- Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE Transactions on Pattern Analysis and Machine Intelligence (2018).
- Nmslib: https://github.com/nmslib/nmslib
