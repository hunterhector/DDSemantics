#!/usr/bin/env bash

mkdir -p ${implicit_corpus}/gigaword_events/hashed_full

for f in ${implicit_corpus}/gigaword_events/nyt_events_shuffled/*.gz
do
    if [[ -f ${f} ]]; then
        h=${f//nyt_events_shuffled/hashed_full}
        echo 'Hashing '${f}
        python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash.py --HashParam.raw_data=${f} --HashParam.output_path=${h}
    fi
done


