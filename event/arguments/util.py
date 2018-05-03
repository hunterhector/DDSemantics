def remove_neg(raw_predicate):
    # Frames of verb with or withour negation should be the same.

    neg = 'not_'
    if raw_predicate.startswith(neg):
        return raw_predicate[len(neg):]

    return raw_predicate
