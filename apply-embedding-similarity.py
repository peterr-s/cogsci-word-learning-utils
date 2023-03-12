#!/usr/bin/env python3

import sys
import argparse
import logging
import csv

from gensim.models import FastText, Word2Vec

logger = logging.getLogger(__name__)
logging.basicConfig(
        format = "[%(levelname)s] %(message)s",
        level = logging.DEBUG
        )

def main(contains_header, word2vec, fasttext, target_column, error_column) :
    # load all models from files
    models = list()
    model_names = list()
    if word2vec is not None :
        for n, path in enumerate(word2vec) :
            models.append(Word2Vec.load(path))
            model_names.append("word2vec %d" % n)
    if fasttext is not None :
        for n, path in enumerate(fasttext) :
            models.append(FastText.load(path))
            model_names.append("fasttext %d" % n)

    # iterate over file, adding columns
    reader = csv.reader(sys.stdin)
    writer = csv.writer(sys.stdout)
    if contains_header :
        header = next(reader)
        header.extend(["%s similarity" % name for name in model_names])
        writer.writerow(header)
    for line in reader :
        target = line[target_column].lower()
        error = line[error_column].lower()
        for n, model in enumerate(models) :
            if target not in model.wv :
                logger.error("target word \"%s\" is not compatible with %s" % (target, model_names[n]))
                sys.exit(1)
            if error not in model.wv :
                logger.warning("error word \"%s\" not found in %s" % (error, model_names[n]))
                line.append(None)
                continue

            similarity = model.wv.similarity(target, error)
            line.append(similarity)
        writer.writerow(line)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
            prog = "apply-embedding-similarity",
            description = "adds csv columns for similarity with the target word of a file"
            )
    parser.add_argument("-c", "--contains-header",
            help = "CSV contains header (default False)",
            default = False,
            type = bool
            )
    parser.add_argument("-w", "--word2vec",
            help = "word2vec artifact location",
            action = "append"
            )
    parser.add_argument("-f", "--fasttext",
            help = "fasttext artifact location",
            action = "append"
            )
    parser.add_argument("-t", "--target-column",
            help = "column number for target word",
            default = 0,
            type = int
            )
    parser.add_argument("-e", "--error-column",
            help = "column number for error word",
            default = 2,
            type = int
            )
    main(**vars(parser.parse_args(sys.argv[1:])))
