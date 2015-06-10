#!/usr/bin/env python
##########################################################
import gensim
import logging
from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from models import LDAModel, LSAModel

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from glove import GloveModel
from util import *#topicSearch, chunkify

from os import listdir
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.svm import SVC

from distributedwordreps import ShallowNeuralNetwork

import logging
import numpy as np
import unittest
import random
from time import ctime

logger = logging.getLogger("cs224u.test")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


def topic_search_example():
    doc = '''
    From: lipman@oasys.dt.navy.mil (Robert Lipman)
Subject: CALL FOR PRESENTATIONS: Navy SciViz/VR Seminar


            CALL FOR PRESENTATIONS
    
      NAVY SCIENTIFIC VISUALIZATION AND VIRTUAL REALITY SEMINAR

            Tuesday, June 22, 1993

        Carderock Division, Naval Surface Warfare Center
          (formerly the David Taylor Research Center)
              Bethesda, Maryland

SPONSOR: NESS (Navy Engineering Software System) is sponsoring a 
one-day Navy Scientific Visualization and Virtual Reality Seminar.  
The purpose of the seminar is to present and exchange information for
Navy-related scientific visualization and virtual reality programs, 
research, developments, and applications.

PRESENTATIONS: Presentations are solicited on all aspects of 
Navy-related scientific visualization and virtual reality.  All 
current work, works-in-progress, and proposed work by Navy 
organizations will be considered.  Four types of presentations are 
available.

     1. Regular presentation: 20-30 minutes in length
     2. Short presentation: 10 minutes in length
     3. Video presentation: a stand-alone videotape (author need not 
    attend the seminar)
     4. Scientific visualization or virtual reality demonstration (BYOH)

Accepted presentations will not be published in any proceedings, 
however, viewgraphs and other materials will be reproduced for 
seminar attendees.

ABSTRACTS: Authors should submit a one page abstract and/or videotape to:

     Robert Lipman
     Naval Surface Warfare Center, Carderock Division
     Code 2042
     Bethesda, Maryland  20084-5000

     VOICE (301) 227-3618;  FAX (301) 227-5753  
     E-MAIL  lipman@oasys.dt.navy.mil

Authors should include the type of presentation, their affiliations, 
addresses, telephone and FAX numbers, and addresses.  Multi-author 
papers should designate one point of contact.

DEADLINES: The abstact submission deadline is April 30, 1993.  
Notification of acceptance will be sent by May 14, 1993.  
Materials for reproduction must be received by June 1, 1993.

For further information, contact Robert Lipman at the above address.

      PLEASE DISTRIBUTE AS WIDELY AS POSSIBLE, THANKS.




Robert Lipman                     | Internet: lipman@oasys.dt.navy.mil
David Taylor Model Basin - CDNSWC |       or: lip@ocean.dt.navy.mil
Computational Signatures and      | Voicenet: (301) 227-3618
   Structures Group, Code 2042    | Factsnet: (301) 227-5753
Bethesda, Maryland  20084-5000    | Phishnet: stockings@long.legs
                   
The sixth sick shiek's sixth sheep's sick.
          '''
    segs, regions = topicSearch(doc, feature_extractor = LDAModel())
    root = parseTree(regions, len(segs))
    # print regions

    for cnt, seg in enumerate(segs):
        print cnt, seg 

    for i in range(10):
        tmp = getLayer(root, i)
        rlt = ''
        for t in tmp:
            rlt += '('+str(t.region[0])+','+str(t.region[1])+'); '
        print rlt

def test():
    sample_size = 20
    decay_iterval = 20

    # load data
    baseFolder = '20news-18828'
    m = LDAModel()
    logfname = 'lda_topK_fine_tune_val.log_'+ctime()
    

    buggyDocs = []
    for depth_iter in range(0, 10):
        for decay_iter in range(0, decay_iterval+1):
            train = []
            trainY = []
            val = []
            valY =[]
            test = []
            testY = []
            opts = {'base_feature_extractor':m, 'depth':depth_iter, 'fullLayer':True, 'decay':decay_iter*1./decay_iterval}
            feature_extractor = TopKLayerHierarchicalFeatureExtractor(opts)
            cats = listdir(baseFolder)
            for catIdx, cat in enumerate(cats):
                # logger.info('Processing category %s (%d/%d)', cat, catIdx, len(cats))
                try:
                    docs = sorted(listdir(os.path.join(baseFolder, cat)), key = int)
                    if sample_size is not None and sample_size != 0:
                        docs = docs[:sample_size]
                except:
                    continue
                numDocs = len(docs)
                for docIdx, doc_filename in enumerate(docs):
                    if (catIdx, docIdx) in buggyDocs:
                        continue
                    try:
                        doc_filename = os.path.join(baseFolder, cat, doc_filename)
                        # logger.info('processing document %s (%d/%d)', doc_filename, docIdx, numDocs)
                        doc = open(doc_filename).read()
                        feature = feature_extractor.featurize(doc)
                        # logger.debug('doc %d feature extracted', docIdx)
                        if docIdx < numDocs*0.8:
                            train.append(feature)
                            trainY.append(catIdx)
                        elif docIdx < numDocs*0.9:
                            val.append(feature)
                            valY.append(catIdx)
                        else:
                            test.append(feature)
                            testY.append(catIdx)
                        # logger.debug('-----')
                        print catIdx, docIdx
                    except:
                        buggyDocs.append((catIdx, docIdx))
                        print 'ERROR:', catIdx, docIdx, doc_filename
                        f = open('error.log', 'a')
                        f.write(str(docIdx)+ ', ' +doc_filename+'\n')
                        f.close()

            # Convert to sparse format for compact storage and minimal memory usage.
            train = np.vstack(train)
            trainY = np.hstack(trainY)
            val = np.vstack(val)
            valY = np.vstack(valY)
            test = np.vstack(test)
            testY = np.hstack(testY)

            # logger.info("Shape of training set: %s", train.shape)
            # logger.info("Shape of test set: %s", test.shape)

            #for clf_class in [GaussianNB, MultinomialNB, LogisticRegression, SVC]:
            for clf_class in [MultinomialNB]:
                classifier_name = function_name(clf_class)
                if classifier_name is None:
                    raise Exception("Unable to get name of classifier class", clf_class)

                logger.info("Evaluating on depth %s, decay %s...", str(depth_iter), str(decay_iter*1./decay_iterval))
                clf = clf_class()
                clf.fit(train, trainY)
                logger.info('training finished')

                # Make prediction.
                valPredY = clf.predict(val)
                testPredY = clf.predict(test)

                # Print detailed report.
                # print(classification_report(testY, testPredY, target_names = cats, digits = 5))
                f = open(logfname, 'a')

                # Save the important metrics for testing.
                precision, recall, f1, support = \
                    precision_recall_fscore_support(testY, testPredY, average='weighted')
                rlt = str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(support)
                print str(depth_iter) + ',' + str(decay_iter*1./decay_iterval) + ':\t' + rlt
                f.write(str(depth_iter) + ',' + str(decay_iter*1./decay_iterval) + ':\t' + rlt + '\n')

                # Save the important metrics for validation
                precision, recall, f1, support = \
                    precision_recall_fscore_support(valY, valPredY, average='weighted')
                rlt = str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(support)
                print str(depth_iter) + ',' + str(decay_iter*1./decay_iterval) + ':\t' + rlt
                f.write(str(depth_iter) + ',' + str(decay_iter*1./decay_iterval) + ':\t' + rlt + '\n')
                f.write('\n')
                f.close()
    # with open(record_fname, "a") as records_out:
    #     json.dump(result_record, records_out, sort_keys = True)
    #     records_out.write("\n")


class NaiveBayesBaseLine(unittest.TestCase):
    def testBaseLine(self):
        #return # disable slow test for now
        logger.info("Running 20NG NB baseline...")

        logger.info("Calculating TF-IDF on 20ng data set...")
        news_train = load_mlcomp('20news-18828', 'train')
        news_test = load_mlcomp('20news-18828', 'test')
        target_names = news_test.target_names
        vectorizer = TfidfVectorizer(encoding='latin1')
        X_train = vectorizer.fit_transform((open(f).read()
                                        for f in news_train.filenames))
        y_train = news_train.target
        X_test = vectorizer.transform((open(f).read() 
                                        for f in news_test.filenames))
        y_test = news_test.target

        del news_train, news_test

        logger.info("Running MultinomialNB...")
        clf = MultinomialNB().fit(X_train, y_train)
        print(classification_report(y_test, clf.predict(X_test),
                                    target_names=target_names))

        del clf

        logger.info("Running pybrain...")

        from pybrain.datasets            import ClassificationDataSet
        from pybrain.utilities           import percentError
        from pybrain.tools.shortcuts     import buildNetwork
        from pybrain.supervised.trainers import BackpropTrainer
        from pybrain.structure.modules   import SoftmaxLayer
        from pybrain.tools.xml.networkwriter import NetworkWriter
        from pybrain.tools.xml.networkreader import NetworkReader

        trndata = ClassificationDataSet(len(vectorizer.get_feature_names()), 1,
                                        nb_classes = len(target_names),
                                        class_labels = target_names)
        for i, x in enumerate(X_train):
            #print x, y_train[i]
            trndata.addSample(x.toarray(), y_train[i])
        trndata._convertToOneOfMany( )
        del X_train, y_train

        tstdata = ClassificationDataSet(len(vectorizer.get_feature_names()), 1,
                                        nb_classes = len(target_names),
                                        class_labels = target_names)
        for i, x in enumerate(X_test):
            tstdata.addSample(x.toarray(), y_test[i])
        tstdata._convertToOneOfMany( )
        del X_test, y_test

        logger.info("Building network...")
        fnn = buildNetwork(trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=0.01,
                                  verbose=True, weightdecay=0.01)

        logger.info("Training pybrain for 50 epochs...")
        trainer.trainEpochs(50)
        pred = fnn.activateOnDataset(tstdata)
        pred = np.argmax(pred, axis=1) # argmax gives the class

        print(classification_report(y_test, pred,
                                    target_names=target_names))


class MockFeatureExtractor(object):
    def num_features(self):
        return 2

    def featurize(self, tokens):
        if random.choice([True, False]):
            return np.array([1, 2], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

class TopicSearchTest(unittest.TestCase):

    def testTopicSearch(self):
        doc = 'This is a doc. I have a clock. Hello Jim. What the duck? '
        feature_extractor = MockFeatureExtractor()
        segments, regions = topicSearch(doc, feature_extractor = feature_extractor)
        logging.info(segments)
        self.assertEqual(4, len(segments))
        for r in regions:
            logging.info(r)
        num_segments = len(segments)
        # one for each merge.
        expected_num_regions = num_segments + (num_segments - 1)
        self.assertEqual(expected_num_regions, len(regions))

    def testChunkify(self):
        s = "This is a long string with no gosh darn punctuation"
        chunks = chunkify(s, 8)
        self.assertGreater(len(chunks), 3, chunks)
        logger.info(chunks)

if __name__ == '__main__':
    topic_search_example()