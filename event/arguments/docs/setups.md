# Implicit Argument Setup
Experiments setups:
1. basic
1. basic_arg_comp_3: using 3 layers for arg composition instead of 2
1. basic_event_comp_3: using 3 layers for event composition instead of 2
1. basic_gaussian_distance: use gaussian to simulate distances
1. basic_biaffine:

Preprocessing training dataset:
1. Parse the large dataset with Stanford and Semafor (for example, get nyt_events.json.gz)
1. Split it into sub files and shuffle the order
    1. ```gunzip -c nyt_all_frames.json.gz | split -l 50000 - nyt_frames_shuffled/part_  --filter='shuf | gzip > $FILE.gz```
1. Creat frame_mapping
    1. ```create_frame_mappings.sh```
        1. ```python -m event.arguments.prepare.frame_collector nyt_events.json.gz frame_maps```
        1. ```python -m event.arguments.prepare.frame_mapper frame_map```
1. Count vocabulary and train event embedding
    1. ```scripts/get_sents_emb_all_frames.sh```
    1. ```scripts/get_sents_emb_event_only.sh```
1. Hash the dataset
    1. ```scripts/hash_train_event_only.sh```
    1. ```scripts/hash_train_all_frames.sh```

Test set setups:
1. Create the automatically constructed training set
    1. Find a domain relevant corpus and parse it with the pipeline
    1. Use the pre-parsed annotated Gigaword NYT portion
1. Obtain the relevant corpus
    1. For G&C Corpus
        1. Read both Propbank and Nombank into the annotations format
        1. Add G&C data into the dataset        
        1. ```python -m event.io.dataset.reader conf/implicit/dataset/gc_data.py```
    1. For SemEval2010 Task 10
        1. Read SemEval dataset into the annotation format
        1. ```python -m event.io.dataset.reader negra```
    1. Run ImplicitFeatureExtractionPipeline to create dataset with features.
        1. Now you will get cloze.json.gz
        1. ```bin/run_pipeline.sh argument-modeling edu.cmu.cs.lti.script.pipeline.ImplicitFeatureExtractionPipeline ~/data/implicit/resources ~/data/implicit/semeval2010t10_train/text ~/data/implicit/semeval2010t10_train/annotation ~/data/implicit/semeval2010t10_train/processed simple framenet```
        1. ```bin/run_pipeline.sh argument-modeling edu.cmu.cs.lti.script.pipeline.ImplicitFeatureExtractionPipeline ~/data/implicit/resources ~/data/implicit/nombank_with_gc/text ~/data/implicit/nombank_with_gc/annotation ~/workspace/implicit/nombank_with_gc/processed simple propbank```
    1. Run hasher to convert it to the Integer format
        1. Use different conf will use different vocab (filter or not) 
        1. ```python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash.py --HashParam.raw_data=cloze.json.gz --HashParam.output_path=cloze_hashed.json.gz```
        1. Or simply ```scripts/hash_test_data.sh```

Data format:
1. The data format for test data is messy because it is a mixture of system 
output and gold standard output, for example:
    1. Front is detected as "Front_for", the actual frame is "Part_orientational"
    1. In the data (json) file, "Front_for" is put at the frame slot, where the 
    actual frame is put at the "eventType" slot.
    