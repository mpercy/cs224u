from util import sentenceSeg, PriorityQueue


def main():
    pass

def topicSearch(doc, model, similarity = cosine, initialPropose = sentenceSeg):
    # facilitating functions
    def getRegion(similarityArray, i, initSeg):
        if similarityArray[i] == 0:
            raise Exception("Not a region head...", "what's this for?")
        j = i
        while(j < len(similarityArray) and similarityArray[j]==0):
            j+=1
        return '.'.join(initSeg[i:j+1])
    def getPrevious(similarityArray, i):
        if i == 0:
             return None
        pre = i-1
        while(similarityArray[i]==0):
               pre -= 1
        return pre
    def getNext(similarityArray, i):
        l = len(similarityArray)
        next = i+1
        while(next<l and similarityArray[next]==0):
            next += 1
        if next >= l:
            return None
        return next

    # initial proposal of regions
    initSeg = initialPropose(doc)
    # recording initial regions
    hypothesesLocations = [(i, i+1) for i in range(len(initSeg))]
    # similarity is recorded as an array the non-zero value is the start of a 
    # segment with its similarity to next
    similaritySet = [0 for _ in range(len(initSeg))]
    # to mark the last region as -1 
    similaritySet[-1] = -1

    # initialize similarity set.
    for i in range(len(similaritySet-1)):
        cur = model(initSeg[i])
	next = model(initSeg[i+1])
	similaritySet[i] = similarity(cur, next)

    while(True):
        # get the most similar
        mostSimilar = np.argmax(similaritySet)
        if similaritySet[mostSimilar] == 0:
            break

        # merge region
        similaritySet[getNext(similaritySet, mostSimilar)] = 0

        # set the similarity score properly
        preIdx = getPrevious(similaritySet, mostSimilar)
        pre = model(getRegion(similaritySet, preIdx, initSeg))
        cur = model(getRegion(similaritySet, mostSimilar, initSeg))
        similaritySet[preIdx] = similarity(pre, cur)
        nxtIdx = getNext(similarSet, mostSimilar)
        if nxtIdx == None:
            similaritySet(mostSimilar) = -1
        else:
            nxt = model(getRegion(similaritySet, nxtIdx, initSeg))
            similaritySet[mostSimilar] = similarity(cur, nxt)

        # add new region to hypotheses locations
        hypothesesLocations.append((mostSimilar, nxtIdx))

    return hypothesesLocations
        


if __name__ == "__main__":
    main()
