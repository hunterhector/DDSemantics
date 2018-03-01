# Objective: Generate Event Mentions with Information from Multiple Source
1. Fully use the small domain specific annotations on event mentions.
1. Expanding the lexical variety by using external corpus.
1. Possibly expanding the ontology by using additional Knowldge Base.

# Main Design:
1. Generation starts from an event type, then predicate, then its arguments.
1. Generate event skeleton (Predicate + Arguments), not full sentence.
1. Learn type-lexical association from domain annotation.
    1. e.g. "Attack.Kill" can be realized by "kill", "assault"
1. Learn lexical-lexical association from external data
    1. Predicate, argument association (Attack with Army)
    1. Argument, argument association (Army and Arms)

# Motivations:
1. Enrich lexical variety through large dataset, such as argument variety
and even predicate variety.
1. Iterative genration-discrimination procedure fine-tunes understanding on
difficult (vauge) event types.
    1. e.g. "Transport Person" or "Transport Artifact".
    1. It is extreme difficult for an classifier to learn the subtle
    differences because only the argument changes, but learning to
    differetiate "Object" and "Human" is not easy.
    1. Consider "smuggle seekers" and "smuggle gold", both only appear in
    the training corpus once, tranditional method would just overfit to
    "gold" or "seekers" as the feature.