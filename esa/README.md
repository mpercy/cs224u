# ESA model

The scripts in this directory implement the cosine-ESA model
( http://en.wikipedia.org/wiki/Explicit_semantic_analysis ).

## Requirements

Software requirements: Python, numpy, scipy, gensim, nltk

I used the following versions (latest as of this writing, obtained via pip):

* numpy (1.9.2)
* scipy (0.15.1)
* gensim (0.11.1-1)
* nltk (3.0.2)

## Download

A pre-trained model with 100K words can be found on Amazon S3.

Download the following files:

* http://mpercy-datasets-01.s3.amazonaws.com/unreasonable.txt (41K)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en_wordids.txt.bz2 (766K)
* http://mpercy-datasets-01.s3.amazonaws.com/wiki_en/wiki_en_bow.mm.index.metadata.cpickle (129M)
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

## Querying the model

The `query_esa.py` script takes an input document and the prefix of the model
(in this case `wiki_en`) in order to perform similarity queries. Example:

```
python query_esa.py unreasonable.txt wiki_en
```

## Building the model

Building the model is pretty straightforward, but it takes quite a few hours,
depending on how many CPUs are available the machine and how fast the disks
are. First, download the Wikipedia dump. Let's say the dump is named
`enwiki-20150403-pages-articles.xml.bz2`

The script to build the model is called `make_esamodel.py` and to see the
options available just run it with the `--help` flag:

```
./make_esamodel.py --help
usage: make_esamodel.py [-h] [--lemmatize] [--dict_size DICT_SIZE]
                        input_file output_prefix

Build an ESA model from a Wikipedia dump.

positional arguments:
  input_file            filename of a bz2-compressed dump of Wikipedia
                        articles, in XML format
  output_prefix         the filename prefix for all output files

optional arguments:
  -h, --help            show this help message and exit
  --lemmatize           use the "pattern" lemmatizer
  --dict_size DICT_SIZE
                        how many of the most frequent words to keep (after
                        removing tokens that appear in more than 10% of all
                        documents). Defaults to 100000

If you have the "pattern" package installed, this script can use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern

Example: ./make_esamodel.py enwiki-20150403-pages-articles.xml.bz2 wiki_en
```
