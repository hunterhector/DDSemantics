#!/usr/bin/env bash

echo "Implicit corpus location is "${implicit_corpus}

echo "Collecting vocabulary and generating sentences"
python -m event.arguments.prepare.event_vocab --input_data ${implicit_corpus}/gigaword_frames/nyt_all_frames.json.gz --vocab_dir ${implicit_corpus}/gigaword_frames/vocab --sent_out ${implicit_corpus}/gigaword_frames/event_sentences/sent

echo "Creating mixed embeddings"
python -m event.arguments.prepare.train_event_embedding "${implicit_corpus}/gigaword_frames/event_sentences/*.gz" "${implicit_corpus}/gigaword_frames/embeddings/event_embeddings_mixed"

echo "Creating frame based embeddings"
python -m event.arguments.prepare.train_event_embedding "${implicit_corpus}/gigaword_frames/event_sentences/sent_with_frames.gz" "${implicit_corpus}/gigaword_frames/embeddings/event_embeddings_with_frame"

echo "Creating predicate based embeddings"
python -m event.arguments.prepare.train_event_embedding "${implicit_corpus}/gigaword_frames/event_sentences/sent_pred_only.gz" "${implicit_corpus}/gigaword_frames/embeddings/event_embeddings_pred_only"
