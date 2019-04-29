# DDSemantics
Learning semantic regularities with data driven approaches

TODO List:
1. The SemEval dataset
    - Check how to map frame args to "arg0" form
1. Adding all dependencies
    - ~~All dependencies are included in training~~
    - Deal with the change of format
1. Refactor the vocab class to make it consistent
    - ~~Refactor the class itself~~
    - Refactor the use cases
1. Embedding baseline result too high
    - Check after removing unk
1. Create an embedding for the empty slot
    - One empty slot vocab
    - Different vocab for different predicate
    - Different vocab for different slot
1. Distance features
    - Distance cutoff baseline
    - Fix the distance embedding error, should deal with infinity much better
1. Classifier for DNI and INI for both datasets.
    - Rule based like Ruppenhofer
    - Classifier like Erk
1. More baselines
    - Using Centering theory baseline
1. Design core experiments: pooling, attention, distance, empty embedding


Experiments setups:
1. basic
1. basic_arg_comp_3: using 3 layers for arg composition instead of 2
1. basic_event_comp_3: using 3 layers for event composition instead of 2
1. basic_gaussian_distance: use gaussian to simulate distances
1. basic_biaffine:

Preprocessing steps:
1. Parse the large dataset with Stanford and Semafor (for example, get nyt_events.json.gz)
1. Split it into sub files 
    1. ```gunzip -c nyt_all_frames.json.gz | split -l 400000 - nyt_frames_shuffled/part_  --filter='gzip > $FILE.gz```
1. Calculate vocabulary
    1. ```python -m event.arguments.prepare.event_vocab --input_data /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/nyt_events.json.gz --vocab_dir /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/vocab --embedding_dir /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/embeddings --sent_out /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/```
    1. This will also create event sentences to train embeddings
1. Train event embedding
    1. ```train_event_vectors```
1. Hash the dataset
    1. make sure ner types are considered
    1. make sure frame types are taken care of 
    1. run hash_train_filter.sh in the scripts directory

Processing steps:
1. Create the automatically constructed training set
    1. Find a domain relevant corpus and parse it with the pipeline
    1. Use the pre-parsed annotated Gigaword NYT portion
1. Obtain the relevant corpus
    1. For G&C Corpus
        1. Read both Propbank and Nombank into the annotations format
        1. Add G&C data into the dataset        
    1. For SemEval2010 Task 10
        1. Read SemEval dataset into the annotation format
        1. ```python -m event.io.dataset.reader negra```
    1. Run ImplicitFeatureExtractionPipeline to create dataset with features.
        1. Now you will get cloze.json.gz
    1. Run hasher to convert it to the Integer format
        1. Use different conf will use different vocab (filter or not) 
        1. ```python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash_filter.py --HashParam.raw_data=cloze.json.gz --HashParam.output_path=cloze_hashed_filter.json.gz```
        1. ```python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash.py --HashParam.raw_data=cloze.json.gz --HashParam.output_path=cloze_hashed.json.gz```



Last TODO List:
1. ~~Reprocess dataset~~
    - ~~Use more lexicon units to enrich the FrameNet output~~
        - ~~Will do if the novel contains unseen frames and stuff~~
    - ~~Add a field for full argument span~~
    - ~~Run this in background~~
        - ~~Now copying data, will process it next~~
    - ~~It is a pity that not all dependencies are added, let's add them back.~~
1. ~~Rehash dataset~~
    - ~~Make the mapping between 1.4 and 1.7~~
    - ~~Mark exact match lexicon as coref~~
1. ~~Remove unknown predicate.~~
    - ~~Use the verb form when reading the data  (Done in generating json)~~
    - ~~Need to regenerate the training data somehow.~~
1. ~~Make sure go over the argument from prepositions.~~
    - ~~check training data (Training data is fine because we use frame parse 
    and dep parse, we haven't use the propbank parse.)~~
    - ~~also check test data~~
1. ~~Use NER type can reduce unknown args~~
    - ~~Convert below threshold args to their ner entity types, else use the original word~~
    - ~~Check why NER tags are not included.~~
1. ~~Include all frames from the training data~~
    - ~~Read all frames~~
    - ~~Take the event frame list and use it in later steps if we want to filter~~
1. The SemEval dataset
    - ~~Parse it~~
    - ~~Do not read "coreference" as frame relations~~
