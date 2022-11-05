import sys
import logging
import argparse

from gensim.models import Word2Vec, FastText, Poincare
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

class CHILDESCorpus(TextCorpus) :
    pass

def main(corpus_path, model_types) :
    logger.info("loading CHILDES")

    # create and train models on provided data
    for model_type in model_types :
        logger.info("creating %s model" % model_type)

        model = None
        if model_type == "word2vec" :
            model = Word2Vec(sentences)
        elif model_type == "fasttext" :
            model = FastText(sentences)
        elif model_type == "poincare" :
            model = Poincare(sentences)

        logger.info("saving model")
        
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
            action = append,
            choices = ["word2vec", "fasttext", "poincare"],
            dest = "model_types",
            help = "the type of model to train",
            required = True
            )

    main(**parser.parse_args(sys.argv[1:]))
