import rdflib
from rdflib import Namespace
from event.util import rm_prefix
import re


class OntologyLoader:
    def __init__(self, ontology_file):
        self.g = rdflib.Graph()
        self.g.load(ontology_file, format='ttl')

        self.ldc_ont = 'ldcOnt'
        self.aida_common = 'aidaCommon'
        self.rdf = 'rdf'
        self.rdfs = 'rdfs'
        self.owl = 'owl'

        self.namespaces = {}
        for prefix, ns in self.g.namespaces():
            self.namespaces[prefix] = Namespace(ns)

        self.prefixes = {}
        self.event_onto = {}
        self.entity_types = set()
        self.filler_types = set()
        self.relation_types = set()
        self.__load()

    def __find_typeof(self, target_type):
        return [a for (a, b, c) in self.g.triples(
            (None, self.namespaces[self.rdf].type, target_type)
        )]

    def __find_values(self, subj):
        return [c for (a, b, c) in self.g.triples(
            (subj, self.namespaces[self.owl].allValuesFrom, None)
        )]

    def shorten(self, uri):
        for prefix, ns in self.namespaces.items():
            if uri.startswith(ns):
                return re.sub('^' + ns, prefix + ':', uri)
        return uri

    def __unpack_list(self, list_node):
        items = set()
        for _, _, res in self.g.triples(
                (list_node, self.namespaces[self.rdf].first, None)):
            items.add(self.shorten(res))

        for _, _, rest in self.g.triples(
                (list_node, self.namespaces[self.rdf].rest, None)):
            items.update(self.__unpack_list(rest))

        return items

    def __get_arg_range(self, arg_role):
        # Load event arguments restrictions.
        restrictions = set()
        for _, _, arg_range in self.g.triples(
                (arg_role, self.namespaces[self.rdfs].subClassOf, None)):
            for arg_range_val in self.__find_values(arg_range):
                for _, _, union_node in self.g.triples(
                        (arg_range_val, self.namespaces[self.owl].unionOf,
                         None)):
                    restrictions.update(self.__unpack_list(union_node))

        return restrictions

    def __load(self):
        # Load entity types.
        for subj in self.__find_typeof(
                self.namespaces[self.aida_common].EntityType):
            self.entity_types.add(self.shorten(subj))

        # Load filler types.
        for subj in self.__find_typeof(
                self.namespaces[self.ldc_ont].FillerType):
            self.filler_types.add(self.shorten(subj))

        # Load relation types.
        for subj in self.__find_typeof(
                self.namespaces[self.aida_common].RelationType):
            self.relation_types.add(self.shorten(subj))

        # Load event types.
        for subj in self.__find_typeof(
                self.namespaces[self.aida_common].EventType):
            # Load event arguments.
            for subj, pred, obj in self.g.triples(
                    (subj, self.namespaces[self.rdfs].subClassOf, None)):
                evm_type = self.shorten(subj)
                if evm_type not in self.event_onto:
                    self.event_onto[evm_type] = {'args': {}}
                for arg_role in self.__find_values(obj):
                    short_arg_role = self.shorten(arg_role)
                    self.event_onto[evm_type]['args'][short_arg_role] = {
                        'restrictions': set()
                    }
                    restrictions = self.__get_arg_range(arg_role)
                    self.event_onto[evm_type]['args'][short_arg_role][
                        'restrictions'].update(restrictions)

    def as_brat_conf(self, conf_path):
        """
        Demonstrate how to convert ontology to a Brat config.
        :return:
        """
        from collections import defaultdict
        grouped_ent_types = defaultdict(list)
        for full_type in self.entity_types:
            onto, t = full_type.split(':')
            grouped_ent_types[onto].append(t)

        grouped_evm_types = defaultdict(list)
        for full_type in self.event_onto.keys():
            onto, t = full_type.split(':')
            grouped_evm_types[onto].append(t)

        with open(conf_path, 'w') as out:
            out.write('[entities]\n\n')

            for onto, types in grouped_ent_types.items():
                out.write('!{}\n'.format(onto))
                for t in types:
                    out.write('\t' + t + '\n')
                out.write('\n')

            out.write('[relations]\n\n')
            for rel_type in self.relation_types:
                out.write(
                    '<OVERLAP>\tArg1:<ENTITY>, Arg2:<ENTITY>, '
                    '<OVL-TYPE>:{}\n'.format(rel_type))
            out.write('\n')

            out.write('[attributes]\n\n')

            out.write('[events]\n\n')
            out.write('#Definition of events.\n\n')
            # out.write('other_event\trelation:<ENTITY>\n')

            for onto, types in grouped_evm_types.items():
                out.write('!{}\n'.format(onto))
                for t in types:
                    full_type = onto + ':' + t

                    out.write('\t' + t)
                    sep = '\t'

                    args = self.event_onto[full_type]['args']
                    for arg in args:
                        plain_arg = arg.split(':')[1]

                        restricts = args[arg]['restrictions']
                        plain_res = [r.split(':')[1] for r in restricts]

                        out.write(
                            '{}{}:{}'.format(sep, plain_arg, '|'.join(plain_res))
                        )
                        sep = ', '
                    out.write('\n')

                # for t, args in type_args.items():
                #     out.write('\t' + t)
                #     sep = '\t'
                #     for arg in args:
                #         out.write('{}{}:<ENTITY>'.format(sep, arg))
                #         sep = ', '
                #     out.write('\n')

    def __find_arg_restrictions(self):
        pass


if __name__ == '__main__':
    loader = OntologyLoader('/home/zhengzhl/projects/data/project_data/'
                            'aida/resources/seedling-ontology.ttl')
    loader.as_brat_conf('temp.conf')
