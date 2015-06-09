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
    max_iters = 10
    error = float('nan')
    old_error = float('nan')
    relative_error_change = float('nan')

    logger.info("Starting k-means clustering with k=%d, max iters=%d, delta=%f", k, max_iters, delta)

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

    logger.info("Preloading memory-mapped shards...")
    for i, shard in enumerate(similarity_index.shards):
        shard.get_index()

    iter = 0
    while iter < max_iters:

        # Calculate cosine similarities between each centroid and each topic.
        # To save time, we also calculate the error for the previous assignment during this step.
        logger.info("Calculating cosine similarity of each cluster with each document...")
        previous_cluster_assignments = np.copy(cluster_assignments)
        previous_cluster_centroids = np.copy(cluster_centroids)
        cluster_counts = np.ones(k) # Use ones instead of zeros to avoid divide by zero.

        cluster_centroids = np.zeros((k, num_terms))
        previous_centroid_distances = np.zeros(k)
        cluster_assignments = []
        docid = 0
        num_shards = len(similarity_index.shards)
        for i, shard in enumerate(similarity_index.shards):
            logger.info("Processing shard %d/%d ...", i, num_shards)
            # Calculate a (Cluster X Document) cosine similarity matrix for the current shard.
            # (C X T) . (T X D) = (C X D)
            logger.info("  Calculating similarities...")
            cluster_shard_similarities = previous_cluster_centroids * shard.get_index().index.transpose()

            # Select most similar cluster for each document.
            logger.info("  Calculating argmax...")
            cluster_selections = np.argmax(cluster_shard_similarities, axis=0)
            cluster_assignments = np.hstack([cluster_assignments, cluster_selections])

            shard_first_docid = docid

            # Calculate errors for the previous assignment.
            # We don't calculate errors on the first iteration since we don't
            # have an assignment yet.
            if previous_cluster_assignments.size != 1: # np.copy() of None has size 1
                logger.info("  Calculating error...")
                for doc_cluster_sims in cluster_shard_similarities.transpose():
                    cluster = previous_cluster_assignments[docid]
                    previous_centroid_distances[cluster] += (1 - doc_cluster_sims[cluster])
                    docid += 1

            # Iteratively recalculate the centroid of each cluster, so we don't
            # have to swap each shard out and back in.
            docid = shard_first_docid # Reset docid counter to before the error calcs.
            logger.info("  Computing new cluster centroids...")
            for topic_vec in shard.get_index().index:
                cluster = cluster_assignments[docid]
                cluster_centroids[cluster] += topic_vec
                cluster_counts[cluster] += 1
                docid += 1

        #print("Cluster assignments:", cluster_assignments)
        cluster_centroids /= cluster_counts[:,None]         # Take the average (off by one to avoid /0)
        cluster_centroids = normalize(cluster_centroids)    # And normalize.

        # We just use the sum of all cosine distances as our error metric.
        old_error = error
        error = np.sum(previous_centroid_distances)
        relative_error_change = abs(1 - error / old_error)

        logger.info("> Iteration: %d, previous error: %f, old error: %f, rel change: %f",
                    iter, error, old_error, relative_error_change)

        # TODO: Drop clusters with zero members assigned and merge clusters that
        # have converged to the same centroid.

        # Checkpoint the clusterings in every iteration so we can test them
        # before they converge.
        # Save centroids.
        centroids_fname = "%s.cluster.%d.centroids" % (model_prefix, k)
        logger.info("Saving clusters to file: %s", centroids_fname)
        s = MatrixSimilarity(None, dtype = np.float64, num_features = num_terms)
        s.index = cluster_centroids
        s.save(centroids_fname)
        del s   # Free any RAM the similarity index might use.

        # Save assignments.
        assignments_fname = "%s.cluster.%d.assignments" % (model_prefix, k)
        logger.info("Saving cluster assignments to file: %s", assignments_fname)
        np.save(open(assignments_fname, 'wb'), cluster_assignments)

        if relative_error_change < delta:
            logger.info("Converged.")
            break

        iter += 1

    logger.info("Done.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s model_prefix" % (sys.argv[0],))
        sys.exit(1)
    k_cluster_wiki(sys.argv[1])
