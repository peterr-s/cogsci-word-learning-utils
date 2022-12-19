#!/usr/bin/env python3

import sys
import os
import logging
import argparse

import nltk
from nltk.corpus.reader import CHILDESCorpusReader

from gensim.models import Word2Vec, FastText #, Poincare
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

class CHILDESCorpus(TextCorpus) :
    def __init__(self) :
        childes_root = nltk.data.find("corpora/childes/data-xml/Eng-USA-MOR/")
        self.reader = CHILDESCorpusReader(childes_root, ".*/.*.xml")

    def get_texts(self) :
        return self.reader.sents(self.reader.fileids(), speaker = ["CHI"])
        #for fileid in self.reader.fileids() :
        #    for sent in self.reader.sents(fileid, speaker = ["CHI"]) :
        #        yield sent

def main(corpus_path, model_types) :
    logger.info("loading CHILDES")
    corpus = CHILDESCorpus()

    # create and train models on provided data
    for model_type in model_types :
        logger.info("creating %s model" % model_type)

        model = None
        if model_type == "word2vec" :
            model = Word2Vec(corpus.get_texts())
        elif model_type == "fasttext" :
            model = FastText(corpus.get_texts())
        #elif model_type == "poincare" :
        #    model = Poincare(corpus.get_texts())

        logger.info("saving model")
        model.save("./model_out")
        
        logger.info("saved model")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
            prog = "train-embedding-models",
            description = "trains various embedding models on the same data"
            )
    parser.add_argument(
            "-c", "--corpus_path",
            help = "path to CHILDES",
            required = True
            )
    parser.add_argument(
            "-m", "--model-type",
            action = "append",
            choices = ["word2vec", "fasttext", "poincare"],
            dest = "model_types",
            help = "the type of model to train",
            required = True
            )

    main(**vars(parser.parse_args(sys.argv[1:])))
