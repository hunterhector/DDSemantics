class NullArgDetector:
    def __init__(self):
        pass

    def should_fill(self, event_info, slot, arg):
        pass


class GoldNullArgDetector(NullArgDetector):
    """A Null arg detector that look at gold standard."""

    def __init__(self):
        super().__init__()

    def should_fill(self, event_info, slot, arg):
        return arg.get('implicit', False) and not arg.get('incorporated', False)


class AllArgDetector(NullArgDetector):
    """A Null arg detector that returns everything."""

    def __init__(self):
        super().__init__()

    def should_fill(self, event_info, slot, arg):
        return True


class ResolvableArgDetector(NullArgDetector):
    """A Null arg detector that returns true for resolvable arguments."""

    def __init__(self):
        super().__init__()

    def should_fill(self, event_info, slot, arg):
        return len(arg) > 0 and arg['resolvable']


class TrainableNullArgDetector(NullArgDetector):
    """A Null arg detector that is trained to predict."""

    def __index__(self):
        super(NullArgDetector, self).__init__()

    def should_fill(self, doc_info, arg_info, arg):
        raise NotImplementedError
