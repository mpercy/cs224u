# code for temporary tasks
from numpy import isnan, array, zeros, argmax
import numpy as np
from esa import ESAModel
from random import randint
import pickle
from gensim.similarities.docsim import MatrixSimilarity
from gensim.matutils import Dense2Corpus



def main():
    k_cluster_wiki()

def k_cluster_wiki():
    k = 2000
    delta = 0.001
    error = float('nan')
    preer = float('nan')

    m = ESAModel('wiki_en-200000--20150531-035019')
    fnum = 10000 #len(m.similarity_index)
    vnum = len(m.dictionary)

    clusterIdcs = [randint(0, k-1) for _ in range(fnum)]
    similarities = MatrixSimilarity(Dense2Corpus(np.array([[3., 2.], [1.,90.]])))
    cnt = 0
    while(isnan(error) or isnan(preer) or abs(1-error/preer)>delta):
        # calculate the cluster centroid
        centroids = [zeros(vnum) for _ in range(k)]
        centCnt = [0. for _ in range(k)]
        print len(centroids), centroids[0].shape
        for i in range(fnum):
            clusterIdx = clusterIdcs[i]
            centroids[clusterIdx] += m.similarity_index.vector_by_id(i)
            centCnt[clusterIdx] += 1.
        print len(centroids), centroids[0].shape
        for i in range(k):
            if centCnt[i] > 0:
                centroids[i] = centroids[i]/centCnt[i]
        centroids = np.vstack(centroids)
        print centroids.shape
        print
        # update topic assigment and error
        preer = error
        error = 0.
        # build a similarity with centroids
        similarities.index = centroids
        # for each shard in m, set the clusterIdcs by comparing the similarity
        shardNum = len(m.similarity_index.shards)
        shardSize = m.similarity_index.shardsize
        batchSize = 10000
        for shardIdx in range(shardNum):
            shard = m.similarity_index.shards[shardIdx]
            batchStart = 0
            while batchStart < shardSize:
                batch = shard.index.index[batchStart:min(batchStart+batchSize,shardSize-1), :]
                clusterSmi = similarities[batch]
                stardIdx = shardIdx*shardSize+batchStart
                endIdx = min(stardIdx+batchSize, len(clusterIdcs))
                clusterIdcs[stardIdx:endIdx] = argmax(clusterSmi, axis=0)
                batchStart += batchSize

        # update error
        for docIdx in range(fnum):
            tmp = np.sequare(centroids[clusterIdcs[docIdx]] - m.similarity_index.vector_by_id(docIdx))
            tmp = sum(tmp)/float(vnum)
            error += tmp
        cnt += 1
        print cnt,  error, preer
    centroids = [zeros(vnum) for _ in range(k)]
    centCnt = [0. for _ in range(k)]
    for i in range(fnum):
        clusterIdx = clusterIdcs[i]
        centroids[clusterIdx] += m.similarity_index.vector_by_id(i)
        centCnt[clusterIdx] += 1.
    centroids = array(centroids)/array(centCnt)

    pickle.dump(centroids, open('clusters.p', wb))


if __name__ == '__main__':
    main()
