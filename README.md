# DDSemantics
Learning semantic regularities with data driven approaches

TODO List:
1. Reprocess dataset
    - Add FrameNet lexicons
    - Make sure prepositions are followed
    - Add a field for full argument span
1. Remove unknown predicate.
    - Use the verb form when reading the data 
    - Use more lexicon units to enrich the FrameNet output 
    - Need to regenerate the training data somehow.
1. Process the SemEval corpus
1. Make sure go over the argument from prepositions.
    - check training data
    - also check test data
1. Use NER type can reduce unknown args
    - Convert below threshold args to their ner entity types
1. Embedding baseline result too high
    - Check after removing unk
1. Consider how to deal with argument pairs that matches
    - Need to be done when regenerating the data.
1. Create an embedding for the empty slot
    - One empty slot vocab
    - Different vocab for different predicate
    - Different vocab for different slot
1. Distance features
    - Distance cutoff baseline
    - Fix the distance embedding error, should deal with infinity much better
1. Classifier for DNI and INI for both datasets.
1. Baselines
    - Embedding baseline result too high.
    - Using Centering theory baseline
1. Design core experiments: pooling, attention, distance, empty embedding


Experiments setups:
1. basic
1. basic_arg_comp_3: using 3 layers for arg composition instead of 2
1. basic_event_comp_3: using 3 layers for event composition instead of 2
1. basic_gaussian_distance: use gaussian to simulate distances
1. basic_biaffine:



