# DDSemantics
Learning semantic regularities with data driven approaches

TODO List:
1. Set up baselines
1. Create an embedding for the empty slot (e.g. One empty slot vocab; 
Different vocab for different predicate; Different vocab for different slot)
1. Use a distance cutoff
1. Classifier for DNI and INI for both datasets.
1. Fix the distance embedding error, should deal with infinity much better.
1. Using Centering theory baseline
1. Using Centering theory as generic filling.
1. Take the SemEval corpus
1. Design core experiments: pooling, attention, distance, empty embedding
1. How to use only the relevant contextual scores?
1. Cheng & Erk use only the head lemma for arguments, shall we use more?
1. Need to be aware that the fiction data used in SemEval2010 may not be
compatible with our training data.

Experiments setups:
1. basic
1. basic_arg_comp_3: using 3 layers for arg composition instead of 2
1. basic_event_comp_3: using 3 layers for event composition instead of 2
1. basic_gaussian_distance: use gaussian to simulate distances
1. basic_biaffine:



