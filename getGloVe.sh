#!/bin/bash
pip3 install flask gensim sklearn matplotlib --user
echo "Downloading GloVe Embeddings ..."
wget http://nlp.stanford.edu/data/glove.6B.zip
echo "Unzipping  the Downloaded ..."
unzip glove.6B.zip -d glove
cd glove
echo "Changing Format ..."
python3 -m gensim.scripts.glove2word2vec --input  glove.6B.50d.txt --output glove.6B.50d.w2v.txt
echo "Move "
mv glove.6B.50d.w2v.txt ../
cd ../
echo "Remove the Download ..."
rm -r glove glove.6B.zip