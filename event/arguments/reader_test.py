from event.arguments.arg_runner import GoldNullArgDetector, ResolvableArgDetector
from event.arguments.data.cloze_readers import HashedClozeReader
from event.arguments.implicit_arg_params import ArgModelPara
from event.arguments.implicit_arg_resources import ImplicitArgResources


class ClozeReaderTester:
    def __init__(self, kwargs):
        self.para = ArgModelPara(**kwargs)
        self.resources = ImplicitArgResources(**kwargs)

        self.reader = HashedClozeReader(self.resources, self.para)
        self.gold_detector = GoldNullArgDetector()
        self.resolvable_detector = ResolvableArgDetector()

    def test_read_test_doc(self, test_lines):
        for test_data in self.reader.read_test_docs(test_lines, self.gold_detector):
            (
                doc_id,
                instances,
                common_data,
                candidate_meta,
                instance_meta,
            ) = test_data


if __name__ == "__main__":
    tester = ClozeReaderTester()
