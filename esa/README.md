# ESA model

The scripts in this directory implement the cosine-ESA model
( http://en.wikipedia.org/wiki/Explicit_semantic_analysis ).

## Requirements

Software requirements: Python, numpy, scipy, gensim

I used the following versions (latest as of this writing, obtained via pip):

* numpy (1.9.2)
* scipy (0.15.1)
* gensim (0.11.1-1)

## Download

A pre-trained model with 100K words can be found on Amazon S3.

Download the following files:

* http://mpercy-datasets-01.s3.amazonaws.com/unreasonable.txt (41K)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en_wordids.txt.bz2 (766K)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en_doc_index.pickle (100M)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en.tfidf_model (4.1M)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en_similarity.tar.gz (3.2G; 4.5G uncompressed)

Then unpack the similarity indexes. This will create 129 files in the current directory:

```
tar xzvf wiki_en_similarity.tar.gz
```

## Prepare documents

The script that queries the model currently expects a text file as input with
one "document" per line. We may want to replace that assumption with something
more useful. An example file in this format is provided as unreasonable.txt

## Query the model

The `query_esa.py` script takes an input document and the prefix of the model
(in this case `wiki_en`) in order to perform similarity queries. Example:

```
python query_esa.py unreasonable.txt wiki_en
```
