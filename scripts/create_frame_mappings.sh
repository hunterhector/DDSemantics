#!/usr/bin/env bash
implicit_corpus=/home/zhengzhl/workspace/implicit

echo "Collect frames first."
python -m event.arguments.prepare.frame_collector ${implicit_corpus}/gigaword_frames/nyt_all_frames.json.gz ${implicit_corpus}/frame_maps

echo "Create mapping int between arguments and frames using the statistics."
python -m event.arguments.prepare.frame_mapper ${implicit_corpus}/frame_maps
