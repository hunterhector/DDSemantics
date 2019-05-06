#!/usr/bin/env bash

if [[ ! -d ${implicit_corpus}/gigaword_events/nyt_events_shuffled ]]; then
    echo 'Spliting into smaller files'
    cd ${implicit_corpus}/gigaword_events/
    mkdir -p ${implicit_corpus}/gigaword_events/nyt_events_shuffled
    gunzip -c  nyt_events.json.gz | split -l 50000 - nyt_events_shuffled/part_ --filter='shuf | gzip > $FILE.gz'
fi

if [[ ! -d ${implicit_corpus}/gigaword_frames/hashed ]]; then
    echo 'Going to do hashing'
    mkdir -p ${implicit_corpus}/gigaword_events/hashed
    cd ~/projects/DDSemantics

    for f in ${implicit_corpus}/gigaword_events/nyt_events_shuffled/*.gz
    do
        if [[ -f ${f} ]]; then
            h=${f//nyt_events_shuffled/hashed}
            echo 'Hashing '${f}' into '${h}
            python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_event_only.py --HashParam.raw_data=${f} --HashParam.output_path=${h}
        fi
    done
fi
