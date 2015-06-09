import logging, gensim, bz2, pickle
numTopics = 7000
def buildLSA():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
	mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
	lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=numTopics)
	pickle.dump(lsi, open('/Volumes/DRIVE/Media/'+str(numTopics)+'_lsa_model.p', 'wb'))

def buildLDA():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
	mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
	lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=numTopics, update_every=1, chunksize=100000, passes=1)
	pickle.dump(lda, open('/Volumes/DRIVE/Media/'+str(numTopics)+'_lda_model.p', 'wb'))

buildLDA()