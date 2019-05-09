# List of TODOs 

1. Richer output in evaluation
1. Study w2v baseline carefully
1. Make sure training reader works properly
1. Get the slot classifier to work
1. Distance features
    - Distance cutoff baseline
    - Fix the distance embedding error, should deal with infinity much better
1. Classifier for DNI and INI for both datasets.
    - Rule based like Ruppenhofer
    - Classifier like Erk
1. Bad idea to use fixed size slots for frame assemble
    - Using a FE-weighted attention should work well
1. More baselines
    - Using Centering theory baseline
1. Create an embedding for the empty slot
    - One empty slot vocab
    - Different vocab for different predicate
    - Different vocab for different slot
1. Design core experiments: pooling, attention, distance, empty embedding

Past TODO List:
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
1. ~~The SemEval dataset~~
    - ~~Check how to map frame args to "arg0" form~~
1. ~~Adding all dependencies~~
    - ~~All dependencies are included in training~~
    - ~~Deal with the change of format~~
1. ~~Refactor the vocab class to make it consistent~~
    - ~~Refactor the class itself~~
    - ~~Refactor the use cases~~
1. ~~UNK_Arguments~~
    - ~~Solve the following ner problem~~
    - ~~If the representing entity is unk, use the text itself~~
1. ~~NER is not correct in the test data~~
    - ~~Example Izquierda Unida~~
    - ~~Solution 1: merge mentions of the same head together and propagate ner type~~
    - ~~Solution 2: allow map to covering entity mention in addition to head word~~
        - ~~Izquierda Unida~~
        - ~~183 years~~
        - ~~hyphenated~~
1. ~~Add correct candidates in test~~
    - ~~try put the gold argument in the ranking~~
    - ~~all other arguments~~
    - ~~all named entities~~
1. ~~Gold standard reading problem to fix~~
1. ~~Embedding baseline result too high~~
    - ~~Check after removing unk~~
