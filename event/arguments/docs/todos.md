# List of TODOs 

1. Bugs:
    1. Training missing key.
    1. Key error when converting the Semeval data.
    1. The test case from the GC data seem to have two problems:
        1. The dep for testing is too simplified (only prep)
        1. Some answers seem to be wrong too.
1. ~~Bug: The same phrase is assigned with different heads and entity id in the
same text, this should be an inconsistent in head finding.~~
    - ~~In: NYT_ENG_20060527.0212, span (2716, 2771) has two mentions: 
    lose or restaurant~~
1. During training, two verbs may share the same exact span, this could be an
 important source of information, but the standard way of creating cloze task
  is to remove the span, hence the information is lost.
1. Baseline, our embedding max method should be at least as strong as the G&C paper 
baseline.
    - ~~Sometimes the system predict "from", this is because the training data 
    contains "IN" as arguments, they should be removed from the generated data.~~
    - ~~Top responses are always the same thing, this may because the same entity
    mention is in multiple arguments' slot, but we should reduce that at 
    evaluation time. The span should be there.~~
    - Multiple results are available in the gold entity, the current result file
     only show the head of the first one, not very informative.
    
1. Richer output in evaluation
    - ~~Add F1~~
    - ~~Add Dice (partial argument overlap)~~
    - ~~Add a score with gold candidates~~
    - ~~Only score candidates with in 2 sentence back and current sentence~~
1. Design core experiments: pooling, attention, distance
1. Distance features
    - Distance cutoff baseline
    - Fix the distance embedding error, should deal with infinity much better
1. ~~Using constituent to find the headword is not reliable, should use dependency~~
    - ~~with having abandoned their socialist principles --> head is with~~
    - the auto cloze data now points to "with", "by", "to", "on", "from"
    - "who" can be pointed to the actual content, in fact, in gold, we sometimes 
    can see "who" being used as the i_arg, but we can find its appositive.
1. analysis through event coref:
    - however, how do we get the correct argument for this first plan?
        - to abandon plans to raise temporary working capital through debt issued from an agency that would n't be counted on the federal budget .
        - another "plan" will take the same arg with this plan.
    - One way is to do carry over via supports.
    - Another way is to use a nombank parser.
    - Semafor did not produce that, we can get it from the "support" (abandon). but that's quite tricky.
    - At the very least, we should get that if the gold candidate is given.
    - Without the gold candidate, we are filling two arguments at the same time
        - fill the first with an close entity
        - fill the second one via coref
1. analysis through the surrounding events:
    - But higher interest rates paid on off-budget debt could add billions to the bailout costs
    - for the costs, we can analyze "bailout" to get the actual entity
        - such as "the new S&L bailout agency"
    - similar analysis should be done on supports
        - The borrowing to raise these funds (raise is the support for funds)
    - Some answers can be directly found in supports 
        - "the RTC may have to slow -LCB- S&L sales -RCB- or dump acquired assets through fire sales"
1. analysis through multiple events:
    - is it possible to read and analyze complex event chains?
        - Du Pont Co. , Hewlett-Packard Co. and Los Alamos National Laboratory 
        said they signed a three-year , $ 11 million agreement to collaborate 
        on superconductor research 
        - Joint-research programs have proliferated as U.S. companies seek to 
        spread the risks and costs of commercializing new superconductors and 
        to meet the challenges posed by foreign consortia
        - signed-subj -> agreement-obj -> collaborate-prep
        - cost-of
    - first, we need to capture all the structures
    - second, we need to have reasoning mechanisms
1. how do we encode the "support" and "surrounding entities"?
    - our training data does not contain these supports anyway
1. Get the slot classifier to work
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
1. Study the influences from context distances
1. The mapping from prep to argN is not one-to-one, how to deal with this?

# Advanced thoughts

1. Removing singleton can help learn INI.
1. Removing entity and its corresponding context can help learning DNI that are 
not resolvable.

# Past TODO List:
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
1. ~~Make sure training reader works properly~~
1. ~~Study w2v baseline carefully~~
1. ~~We are using the detailed dependency labels~~
1. ~~Setup google platform~~
1. ~~Dependency~~
    - ~~Rerun the parser with the old dependency maybe?~~
