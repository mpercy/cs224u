# ESA model

The scripts in this directory implement the cosine-ESA model
( http://en.wikipedia.org/wiki/Explicit_semantic_analysis ).

## Requirements

Software requirements: Python, numpy, scipy, gensim, nltk, scikit-learn

I used the following versions (latest as of this writing, obtained via pip):

* numpy (1.9.2)
* scipy (0.15.1)
* nltk (3.0.2)
* scikit-learn (0.16.1)

At the time of this writing, a custom version of `gensim` is required.
Install it with `pip` using the following incantation:

```
pip install https://github.com/mpercy/gensim/archive/reverse-indexes-2.zip
```

## Download

A pre-trained model with a reverse index and a 200K word vocabulary can be found on Amazon S3.

Use the `aws` command-line tool from Amazon to sync the following S3 directory (about 30G):

s3://mpercy-datasets-01/wiki_en-200000--20150531-035019/

This will create about 300 files in the current directory.

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
