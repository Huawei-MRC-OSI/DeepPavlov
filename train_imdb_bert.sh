#!/bin/sh

export DP_CACHE_DIR=$HOME/.deeppavlov/cache

# Probably run this first (make sure that you are in a virtual env)
# python -m deeppavlov install deeppavlov/configs/classifiers/sentiment_imdb_bert.json

python -u -m deeppavlov train deeppavlov/configs/classifiers/sentiment_imdb_bert.json -d
python -u -m deeppavlov train deeppavlov/configs/classifiers/sentiment_imdb_conv_bert.json -d

# The trained models will be at ~/.deeppavlov/models/classifiers/sentiment_imdb_{,conv_}bert_v0
# I suppose that we need to simply tar these two directories. Or maybe remove train valid logs and
# the checkpoint file and then tar.
