#!/usr/bin/env bash

if [[ ! -d ${implicit_corpus}/gigaword_frames/nyt_all_frames_shuffled ]]; then
    echo 'Spliting into smaller files'
    cd ${implicit_corpus}/gigaword_frames/
    mkdir -p ${implicit_corpus}/gigaword_frames/nyt_all_frames_shuffled
    gunzip -c nyt_all_frames.json.gz | split -l 50000 - nyt_all_frames_shuffled/part_ --filter='shuf | gzip > $FILE.gz'
fi

hash_dir='hashed_new'

echo 'Going to do hashing'
mkdir -p ${implicit_corpus}/gigaword_frames/${hash_dir}
cd ~/projects/DDSemantics

for f in ${implicit_corpus}/gigaword_frames/nyt_all_frames_shuffled/*.gz
do
    if [[ -f ${f} ]]; then
        h=${f//nyt_all_frames_shuffled/${hash_dir}}

        if [[ -f ${h} ]]; then
            echo ${h}' already exists'
        else
            echo 'Hashing '${f}' into '${h}
            python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_all_frames.py --HashParam.raw_data=${f} --HashParam.output_path=${h}
        fi
    fi
done

