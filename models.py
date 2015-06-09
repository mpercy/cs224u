# inlcudes different models to convert a text segmentation into an array
from nltk.tokenize import wordpunct_tokenize
from gensim.corpora import Dictionary
import pickle, gensim, logging
from scipy.sparse import coo_matrix

class LDAModel:
	def __init__(self, model_prefix = 'wiki_en'):
		logger = logging.getLogger("LDA")
		self.model_prefix = model_prefix
		if self.model_prefix is None:
			raise ValueError("model_prefix must be specified")

		self.fname = 'lda_model.p'

		logger.info("LDA: Loading word dictionary...")
		self.dict = Dictionary.load_from_text(model_prefix + '_wordids.txt')

		logger.info("LDA: Loading pretrained model...")
		self.model = pickle.load(open(self.fname, 'r'))

		logger.info("LDA: Finished loading model files.")

	def num_features(self):
		return self.model.num_topics

	def num_terms(self):
		return self.model.num_terms

	def featurize(self, input_str):
		input_str = gensim.utils.to_utf8(input_str, errors='replace').decode("utf8")
		doc = wordpunct_tokenize(input_str)
		doc = [w.lower() for w in doc]

		# Convert from tokens to word ids from the model dictionary.
		doc_bow = self.dict.doc2bow(doc)

		# Simply add up all the vectors and return.
		vec = self.model[doc_bow]
		col = []
		data = []
		for topicNum, value in vec:
			data.append(value)
			col.append(topicNum)

		row = [0 for _ in range(len(data))]
		vec = coo_matrix((data, (row, col)), shape = (1, self.model.num_topics)).toarray()
		return vec


class LSAModel:
	def __init__(self, model_prefix = 'wiki_en'):
		logger = logging.getLogger("LSA")
		self.model_prefix = model_prefix
		if self.model_prefix is None:
			raise ValueError("model_prefix must be specified")

		self.fname = 'lsa_model.p'

		logger.info("LSA: Loading word dictionary...")
		self.dict = Dictionary.load_from_text(model_prefix + '_wordids.txt')

		logger.info("LSA: Loading pretrained model...")
		self.model = pickle.load(open(self.fname, 'r'))

		logger.info("LSA: Finished loading model files.")

	# def __init__(self):
	# 	self.fname = 'lsa_model.p'
	# 	self.model = pickle.load(open(self.fname, 'r'))
	# 	self.dict = Dictionary.load_from_text('wiki_en_wordids.txt')

	def num_features(self):
		return self.model.num_topics

	def num_terms(self):
		return self.model.num_terms

	def featurize(self, input_str):
		input_str = gensim.utils.to_utf8(input_str, errors='replace').decode("utf8")
		doc = wordpunct_tokenize(input_str)
		doc = [w.lower() for w in doc]

		# Convert from tokens to word ids from the model dictionary.
		doc_bow = self.dict.doc2bow(doc)

		# Simply add up all the vectors and return.
		vec = self.model[doc_bow]
		col = []
		data = []
		for topicNum, value in vec:
			data.append(value)
			col.append(topicNum)

		row = [0 for _ in range(len(data))]
		vec = coo_matrix((data, (row, col)), shape = (1, self.model.num_topics)).toarray()
		return vec