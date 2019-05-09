#!/usr/bin/env bash
implicit_corpus=/home/zhengzhl/workspace/implicit

#echo "Collecting vocabulary and generating sentences."
#python -m event.arguments.prepare.event_vocab --input_data ${implicit_corpus}/gigaword_events/nyt_events.json.gz --vocab_dir  ${implicit_corpus}/gigaword_events/vocab --sent_out ${implicit_corpus}/gigaword_events/event_sentences/sent

echo "Creating frame based embeddings"
python -m event.arguments.prepare.train_event_embedding ${implicit_corpus}/gigaword_events/event_sentences/sent_with_frames.gz ${implicit_corpus}/gigaword_events/embeddings/event_embeddings_with_frame

echo "Creating predicate based embeddings"
python -m event.arguments.prepare.train_event_embedding ${implicit_corpus}/gigaword_events/event_sentences/sent_pred_only.gz ${implicit_corpus}/gigaword_events/embeddings/event_embeddings_with_pred_only

#echo "Creating mixed embeddings"
#python -m event.arguments.prepare.train_event_embedding "${implicit_corpus}/gigaword_events/event_sentences/*.gz" ${implicit_corpus}/gigaword_events/embeddings/event_embeddings_mixed

