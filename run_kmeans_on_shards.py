#!/usr/bin/env python
############################################################################
# Run k-means clustering on a forward similarity index to reduce the number
# of documents in the index.

import logging
import numpy as np
import sys

from esa import ESAModel
from gensim.similarities import MatrixSimilarity
from sklearn.preprocessing import normalize

logger = logging.getLogger("k-means")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

def k_cluster_wiki(model_prefix):
    k = 2000
    delta = 0.001
    max_iters = 100
    error = float('nan')
    old_error = float('nan')
    relative_error_change = float('nan')

    logger.info("Starting k-means clustering with k=%d", k)

    m = ESAModel(model_prefix)
    similarity_index = m.similarity_index
    dictionary = m.dictionary

    num_topics = len(similarity_index)
    num_terms = len(dictionary)

    # Create initial cluster centroids.
    # L2-normalize them so we can calculate cosine similarity with a simple dot product.
    cluster_centroids = normalize(np.random.uniform(size=(k, num_terms)))

    # The cluster that each document belongs to.
    cluster_assignments = None

    iter = 0
    while iter < max_iters:

        # Calculate cosine similarities between each centroid and each topic.
        # To save time, we also calculate the error for the previous assignment during this step.
        logger.info("Calculating cosine similarity of each cluster with each document...")
        previous_cluster_assignments = cluster_assignments
        previous_centroid_distances = np.zeros(k)
        cluster_assignments = []
        docid = 0
        for shard in similarity_index.shards:
            # Calculate a (Cluster X Document) cosine similarity matrix for the current shard.
            # (C X T) . (T X D) = (C X D)
            cluster_shard_similarities = cluster_centroids * shard.get_index().index.transpose()

            # Select most similar cluster for each document.
            cluster_selections = np.argmax(cluster_shard_similarities, axis=0)
            cluster_assignments.append(cluster_selections)

            # Calculate errors for the previous assignment.
            # We don't calculate errors on the first iteration since we don't
            # have an assignment yet.
            if previous_cluster_assignments is not None:
                for doc_cluster_sims in cluster_shard_similarities.transpose():
                    cluster = previous_cluster_assignments[docid]
                    previous_centroid_distances[cluster] += (1 - doc_cluster_sims[cluster])
                    docid += 1

        cluster_assignments = np.hstack(cluster_assignments)
        #print("Cluster assignments:", cluster_assignments)

        # We just use the sum of all cosine distances as our error metric.
        old_error = error
        error = np.sum(previous_centroid_distances)
        relative_error_change = abs(1 - error / old_error)

        # Recalculate the centroid of each cluster.
        logger.info("Recalculating the centroid of each cluster...")
        cluster_centroids = np.zeros((k, num_terms))
        cluster_counts = np.ones(k)
        docid = 0
        for shard in similarity_index.shards:
            for doc_term_vector in shard.get_index().index.toarray():
                cluster = cluster_assignments[docid]
                cluster_centroids[cluster] += doc_term_vector
                cluster_counts[cluster] += 1
                docid += 1
        cluster_centroids /= cluster_counts[:,None]         # Take the average (off by one to avoid /0)
        cluster_centroids = normalize(cluster_centroids)    # And normalize.

        logger.info("Iteration: %d, error: %f, previous error: %f, rel change: %f",
                    iter, error, old_error, relative_error_change)
        if relative_error_change < delta:
            logger.info("Converged.")
            break

        iter += 1

    # TODO: Drop clusters with zero members assigned and merge clusters that
    # have converged to the same centroid.

    centroids_fname = "%s.cluster.%d.centroids" % (model_prefix, k)
    logger.info("Saving clusters to file: %s", centroids_fname)
    s = MatrixSimilarity(None, dtype = np.float64, num_features = num_terms)
    s.index = cluster_centroids
    s.save(centroids_fname)

    assignments_fname = "%s.cluster.%d.assignments" % (model_prefix, k)
    logger.info("Saving cluster assignments to file: %s", assignments_fname)
    np.save(open(assignments_fname, 'wb'), cluster_assignments)

    logger.info("Done.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s model_prefix" % (sys.argv[0],))
        sys.exit(1)
    k_cluster_wiki(sys.argv[1])
