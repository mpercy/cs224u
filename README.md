# cs224u
cs224u

## Working with the Glove model

1. Download the "Wikipedia 2014 + Gigaword 5 (6B tokens)" 300d glove dataset from http://nlp.stanford.edu/projects/glove/  
   Note: the original file is named .gz but is not compressed, you may want to just rename it to .txt
2. Convert into numpy format via: `./make_glove_dict.py glove.6B.300d.txt`
3. Run the evaluation: `./main.py glove.6B.300d.txt ./20news-18828`
