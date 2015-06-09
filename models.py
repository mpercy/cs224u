# inlcudes different models to convert a text segmentation into an array
from nltk.tokenize import wordpunct_tokenize

class LDAModel:
	def __init__(self):
		self.fname = 'lda_model.p'
		self.model = pickle.load(open(self.fname, 'r'))
		self.dictionary = Dictionary.load_from_text('wiki_en_wordids.txt')

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
		return vec


class LSALModel:
	def __init__(self):
		self.fname = 'lsa_model.p'
		self.model = pickle.load(open(self.fname, 'r'))
		self.dictionary = Dictionary.load_from_text('wiki_en_wordids.txt')

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
		return vec