#!/usr/bin/env bash
implicit_corpus=/home/zhengzhl/workspace/implicit

echo "hash nombank"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_event_only.py --HashParam.raw_data=${implicit_corpus}/nombank_with_gc/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/nombank_with_gc/processed/cloze_hashed.json.gz
echo "hash semeval train"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_all_frames.py --HashParam.raw_data=${implicit_corpus}/semeval2010t10_train/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/semeval2010t10_train/processed/cloze_hashed.json.gz
echo "hash semeval test"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_all_frames.py --HashParam.raw_data=${implicit_corpus}/semeval2010t10_test/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/semeval2010t10_test/processed/cloze_hashed.json.gz
