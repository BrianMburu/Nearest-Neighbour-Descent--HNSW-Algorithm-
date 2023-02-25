import nmslib
import time


class HNSW:
    """
    NN-Descent, HNSW algorithm
    """

    def __init__(self, index=None, print_progress=False):
        """
        Initialize an instance of the HNSW class.

        Args:
        index: An optional pre-built index for the HNSW algorithm.
        print_progress: A boolean indicating whether or not to print progress messages during initialization.
        """
        self.print_progress = print_progress
        self.index = index

    def initialize(self, known_embeddings, post=2):
        """
        Build an index for the HNSW algorithm based on known embeddings.

        Args:
        known_embeddings: A list of known embeddings to build the index on.
        post: A parameter that controls the trade-off between index build time and query time.

        Returns:
        None
        """
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(known_embeddings)
        self.index.createIndex({'post': post}, self.print_progress)

    def get_closest_neighbour(self, query_embedding, known_labels):
        """
        Find the closest neighbor to a query embedding in the HNSW index.

        Args:
        query_embedding: The embedding to find the closest neighbor for.
        known_labels: A list of known labels corresponding to the embeddings used to build the index.

        Returns:
        A tuple containing the closest label and distance.
        """
        closest_index, closest_distance = self.index.knnQuery(query_embedding, k=1)
        closest_label = known_labels[closest_index[0]]
        return closest_label, closest_distance[0]

    def test_performance(self, testX, testy, known_labels):
        """
        Test the performance of the HNSW algorithm on a test dataset.

        Args:
        testX: A list of test embeddings.
        testy: A list of labels corresponding to the test embeddings.
        known_labels: A list of known labels corresponding to the embeddings used to build the index.

        Returns:
        A tuple containing the accuracy, number of mislabeled examples, and total time taken for testing.
        """
        mis_labeled = 0
        start_time = time.time()

        for i, _ in enumerate(testX):
            closest_label, _ = self.get_closest_neighbour(testX[i], known_labels)
            if testy[i] != closest_label:
                mis_labeled += 1

        end_time = time.time()
        total_time = end_time - start_time
        accuracy = 100 - (mis_labeled / len(known_labels))

        return accuracy, mis_labeled, total_time
