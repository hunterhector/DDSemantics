#!/usr/bin/env bash

mkdir -p ${implicit_corpus}/gigaword_frames/hashed

for f in ${implicit_corpus}/gigaword_frames/nyt_all_frames_shuffled/*.gz
do
    if [[ -f ${f} ]]; then
        h=${f//nyt_events_shuffled/hashed}
        echo 'Hashing '${f}' into '${h}
        python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash.py --HashParam.raw_data=${f} --HashParam.output_path=${h}
    fi
done

