#!/user.bin/env python
# this script trains a lda model for 20 news group for classification
# the result is
#                           precision    recall  f1-score   support

#              alt.atheism    0.00000   0.00000   0.00000         6
#            comp.graphics    0.22727   0.10309   0.14184        97
#  comp.os.ms-windows.misc    0.20000   0.20408   0.20202        98
# comp.sys.ibm.pc.hardware    0.29167   0.35714   0.32110        98
#    comp.sys.mac.hardware    0.37500   0.12500   0.18750        96
#           comp.windows.x    0.34667   0.53061   0.41935        98
#             misc.forsale    0.55952   0.48454   0.51934        97
#                rec.autos    0.45349   0.39394   0.42162        99
#          rec.motorcycles    0.28025   0.44444   0.34375        99
#       rec.sport.baseball    0.33803   0.48485   0.39834        99
#         rec.sport.hockey    0.42667   0.32323   0.36782        99
#                sci.crypt    0.20874   0.43434   0.28197        99
#          sci.electronics    0.32500   0.26531   0.29213        98
#                  sci.med    0.40000   0.10101   0.16129        99
#                sci.space    0.29885   0.26531   0.28108        98
#   soc.religion.christian    0.14433   0.56566   0.22998        99
#       talk.politics.guns    0.28571   0.04396   0.07619        91
#    talk.politics.mideast    0.84615   0.11702   0.20561        94
#       talk.politics.misc    0.00000   0.00000   0.00000        77
#       talk.religion.misc    0.00000   0.00000   0.00000        62

#              avg / total    0.32393   0.28564   0.26356      1803
import logging
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
import gensim

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from nltk.tokenize import wordpunct_tokenize
from scipy.sparse import coo_matrix
from collections import defaultdict

from os import listdir
import os, pickle, numpy

logger = logging.getLogger("cs223u.baseline")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
sample_size = None
def load_texts():
	train = []
	trainY = []
	test = []
	testY = []
	frequency = defaultdict(int)
	basedir = '20news-18828'
	cats = listdir(basedir)

	# iterate through all texts
	for catIdx, cat in enumerate(cats):
		try:
			docs = sorted(listdir(os.path.join(basedir, cat)), key = int)
			if sample_size is not None and sample_size != 0:
				docs = docs[:sample_size]
		except:
			continue
		numDocs = len(docs)

		for docIdx, doc_filename in enumerate(docs):
			doc_filename = os.path.join(basedir, cat, doc_filename)
			# logger.info('processing document %s (%d/%d)', doc_filename, docIdx, numDocs)
			doc = open(doc_filename).read()
			doc = gensim.utils.to_utf8(doc, errors='replace').decode("utf8")
			doc = wordpunct_tokenize(doc)
			doc = [w.lower() for w in doc]

			for w in doc:
				frequency[w] += 1
			if docIdx < 0.9*numDocs:
				train.append(doc)
				trainY.append(catIdx)
			else:
				test.append(doc)
				testY.append(catIdx)
			# logger.debug('doc %d feature extracted', docIdx)

	# trim low freqeucy words
	train = [[w for w in text if frequency[w] > 1 ] for text in train]
	test = [[w for w in text if frequency[w] > 1 ] for text in test]

	# build vocabulary dict
	dictionary = corpora.Dictionary(train)


	train = [dictionary.doc2bow(t) for t in train]
	test = [dictionary.doc2bow(t) for t in test]

   	return train, trainY, test, testY, dictionary, cats



def main():
	# trains lda model
	train, trainY, test, testY, word_dict, cats = load_texts()
	try:
		lda = pickle.load(open('lda_baseline_model.p'))
		print 'LDA model loaded...'
	except:
		lda = LdaModel(train, num_topics=100)
		pickle.dump(lda, open('lda_baseline_model.p', 'wb'))

	# transform training data
	tmpT = []
	for t in train:
		vec = lda[t]
		col = []
		data = []
		for topicNum, value in vec:
			data.append(value)
			col.append(topicNum)

		row = [0 for _ in range(len(data))]
		vec = coo_matrix((data, (row, col)), shape = (1, lda.num_topics)).toarray()
		tmpT.append(vec)
	tmpT = numpy.vstack(tmpT)
	print len(train), tmpT.shape

	# trains classifier
	cls = MultinomialNB()
	cls.fit(tmpT, trainY)

	# transform test data
	tmpT = []
	for t in test:
		vec = lda[t]
		col = []
		data = []
		for topicNum, value in vec:
			data.append(value)
			col.append(topicNum)

		row = [0 for _ in range(len(data))]
		vec = coo_matrix((data, (row, col)), shape = (1, lda.num_topics)).toarray()
		tmpT.append(vec)
	tmpT = numpy.vstack(tmpT)
	predicted = cls.predict(tmpT)

	print(classification_report(testY, predicted, target_names = cats, digits = 5))

	# evaluation
	pass



if __name__ == '__main__':
	main()