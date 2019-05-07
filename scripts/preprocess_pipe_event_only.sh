#!/usr/bin/env bash

cd ~/projects/DDSemantics
implicit_corpus=/home/zhengzhl/workspace/implicit

echo "Running the embedding script"
scripts/get_sents_emb_event_only.sh

echo "Running the hashing script"
scripts/hash_train_event_only.sh
