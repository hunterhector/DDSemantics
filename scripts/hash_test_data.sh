#!/usr/bin/env bash
if [[ -z "${implicit_corpus}" ]]; then
  echo 'Env variable implicit_corpus not defined'
  exit
fi

echo "hash nombank"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_event_only.py --HashParam.raw_data=${implicit_corpus}/nombank_with_gc/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/nombank_with_gc/processed/cloze_hashed.json.gz --HashParam.strict_arg_count=True
echo "hash semeval train"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_all_frames.py --HashParam.raw_data=${implicit_corpus}/semeval2010t10_train/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/semeval2010t10_train/processed/cloze_hashed.json.gz --HashParam.strict_arg_count=True
echo "hash semeval test"
 python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_all_frames.py --HashParam.raw_data=${implicit_corpus}/semeval2010t10_test/processed/cloze.json.gz --HashParam.output_path=${implicit_corpus}/semeval2010t10_test/processed/cloze_hashed.json.gz --HashParam.strict_arg_count=True
