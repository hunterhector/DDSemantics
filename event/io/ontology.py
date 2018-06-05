import rdflib


class OntologyLoader:
    def __init__(self, ontology_file):
        self.g = rdflib.Graph()
        self.g.load(ontology_file, format='ttl')

    def test(self):
        import pprint
        for stmt in self.g:
            pprint.pprint(stmt)


if __name__ == '__main__':
    loader = OntologyLoader('/home/zhengzhl/projects/data/project_data/'
                            'aida/resources/seedling-ontology.ttl')
    loader.test()
