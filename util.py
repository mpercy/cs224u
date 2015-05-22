import scipy

sentenceEnds = ['...', '.', '.', '!', '?']

def sentenceSeg(doc):
    # new paragraph is meaningless here
    doc = doc.replace('\n', '').replace('\r', '')
    # split the doc with sentence ending marks
    initialRegions = [doc]
    for sentenceEnd in sentenceEnds:
        tmp = []
        for region in initialRegions:
            tmp += region.split(sentenceEnd)
        initialRegions = tmp
    return [x for x in initialRegions if x!='']


def cosine(x, y):
    return scipy.spatial.distance.cosine(u, v)


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def isEmpty():
        return self._queue==[]
