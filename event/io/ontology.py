import rdflib
from rdflib import Namespace
import re
import logging


class OntologyLoader:
    def __init__(self, ontology_file):
        logging.info("Loading ontology from : {}".format(ontology_file))

        self.g = rdflib.Graph()
        self.g.load(ontology_file, format='ttl')

        self.ldc_ont = 'ldcOnt'
        self.aida_common = 'aidaCommon'
        self.rdf = 'rdf'
        self.rdfs = 'rdfs'
        self.owl = 'owl'
        self.schema = 'schema'

        self.namespaces = {}
        for prefix, ns in self.g.namespaces():
            self.namespaces[prefix] = Namespace(ns)

        self.prefixes = {}
        self.event_onto = {}
        self.entity_types = set()
        self.filler_types = set()
        self.relation_types = set()
        self.labels = {}
        self.__load()

    def __find_subclassof(self, target_class):
        return [a for (a, b, c) in self.g.triples(
            (None, self.namespaces[self.rdfs].subClassOf, target_class)
        )]

    def __find_values(self, subj):
        return [c for (a, b, c) in self.g.triples(
            (subj, self.namespaces[self.owl].allValuesFrom, None)
        )]

    def shorten(self, uri):
        for prefix, ns in self.namespaces.items():
            if uri.startswith(ns):
                return prefix, ns, re.sub('^' + ns, '', uri)
        return '', '', uri

    def __unpack_list(self, list_node):
        items = set()
        for _, _, res in self.g.triples(
                (list_node, self.namespaces[self.rdf].first, None)):
            items.add(res)

        for _, _, rest in self.g.triples(
                (list_node, self.namespaces[self.rdf].rest, None)):
            items.update(self.__unpack_list(rest))

        return items

    def __get_arg_range(self, arg_role):
        # Load event arguments restrictions.
        restrictions = set()
        for _, _, arg_range in self.g.triples(
                (arg_role, self.namespaces[self.schema].rangeIncludes, None)):
            restrictions.add(arg_range)
        return restrictions

    def __load(self):
        # Load entity types.
        for subj in self.__find_subclassof(
                self.namespaces[self.aida_common].EntityType):
            self.entity_types.add(subj)

        # Load filler types.
        for subj in self.__find_subclassof(
                self.namespaces[self.ldc_ont].FillerType):
            self.filler_types.add(subj)

        # Load relation types.
        for subj in self.__find_subclassof(
                self.namespaces[self.aida_common].RelationType):
            self.relation_types.add(subj)

        # Load event types.
        for evm_type in self.__find_subclassof(
                self.namespaces[self.aida_common].EventType):
            if evm_type not in self.event_onto:
                self.event_onto[evm_type] = {'args': {}}

        arg_events = {}
        # Load event arguments.
        for arg_type in self.__find_subclassof(
                self.namespaces[self.aida_common].EventArgumentType
        ):
            for _, _, evm_type in self.g.triples(
                    (arg_type, self.namespaces[self.rdfs].domain, None)):
                self.event_onto[evm_type]['args'][arg_type] = {}
                arg_events[arg_type] = evm_type

        for arg_type, evm_type in arg_events.items():
            for _, _, restrictions in self.g.triples((
                    arg_type, self.namespaces[self.schema].rangeIncludes, None
            )):
                self.event_onto[evm_type]['args'][arg_type][
                    'restrictions'] = set()
                restrictions = self.__get_arg_range(arg_type)
                self.event_onto[evm_type]['args'][arg_type][
                    'restrictions'].update(restrictions)

            for _, _, label in self.g.triples((
                    arg_type, self.namespaces[self.rdfs].label, None
            )):
                self.event_onto[evm_type]['args'][arg_type][
                    'label'] = label

        # Load labels.
        for origin, _, label in self.g.triples((
                None, self.namespaces[self.rdfs].label, None
        )):
            print(origin)
            if origin == self.namespaces[self.ldc_ont]:
                self.labels[origin] = label

    def as_text(self, output_path):
        with open(output_path, 'w') as out:
            print(len(self.event_onto))
            for full_type, args in self.event_onto.items():
                print(full_type)
                out.write(full_type)
                out.write('\t')
                out.write(' '.join(args))

    def as_brat_conf(self, conf_path, visual_path=None):
        """
        Demonstrate how to convert ontology to a Brat config.
        :return:
        """
        from collections import defaultdict
        grouped_ent_types = defaultdict(list)
        for full_type in self.entity_types:
            prefix, ns, short = self.shorten(full_type)
            grouped_ent_types[prefix].append((short, full_type))

        grouped_filler_types = defaultdict(list)
        for full_type in self.filler_types:
            prefix, ns, short = self.shorten(full_type)
            grouped_filler_types[prefix].append((short, full_type))

        grouped_evm_types = defaultdict(list)
        for full_type in self.event_onto.keys():
            prefix, ns, short = self.shorten(full_type)
            grouped_evm_types[prefix].append((short, full_type))

        short_relation_types = []
        for full_type in self.relation_types:
            prefix, ns, short = self.shorten(full_type)
            short_relation_types.append(short)

        with open(conf_path, 'w') as out:
            out.write('[entities]\n\n')

            for onto, types in grouped_ent_types.items():
                out.write('!{}\n'.format(onto + '_entity'))
                for t, full_type in sorted(types):
                    out.write('\t' + t + '\n')
                out.write('\n')

            for onto, types in grouped_filler_types.items():
                out.write('!{}\n'.format(onto + '_filler'))
                for t, full_type in sorted(types):
                    out.write('\t' + t + '\n')
                out.write('\n')

            out.write('[relations]\n\n')
            for rel_type in sorted(short_relation_types):
                out.write(
                    '{}\tArg1:<ENTITY>, Arg2:<ENTITY>'
                    '\n'.format(rel_type))
                out.write(
                    '<OVERLAP>\tArg1:<ENTITY>, Arg2:<ENTITY>, '
                    '<OVL-TYPE>:<ANY>\n'
                )
            out.write('\n')

            out.write('[attributes]\n\n')

            out.write('[events]\n\n')
            out.write('#Definition of events.\n\n')

            for onto, types in grouped_evm_types.items():
                out.write('!{}\n'.format(onto + '_event'))
                for t, full_type in sorted(types):
                    out.write('\t' + t)
                    sep = '\t'

                    args = self.event_onto[full_type]['args']
                    for arg, arg_content in args.items():
                        arg_prefix, arg_ns, arg_type = self.shorten(arg)

                        restricts = arg_content['restrictions']
                        plain_res = []

                        for r in restricts:
                            _, _, restrict_type = self.shorten(r)
                            plain_res.append(restrict_type)

                        out.write(
                            '{}{}:{}'.format(sep, arg_type,
                                             '|'.join(plain_res))
                        )
                        sep = ', '
                    out.write('\n')

        if visual_path:
            with open(visual_path, 'w') as out:
                out.write('[labels]\n\n')
                for origin, label in self.labels.items():
                    prefix, ns, t = self.shorten(origin)
                    out.write(t + ' | ' + label)
                    if len(t) < len(label):
                        out.write(' | ' + t)
                    out.write('\n')

                out.write('\n[drawing]\n')

    def __find_arg_restrictions(self):
        pass


if __name__ == '__main__':
    from event import util

    util.set_basic_log()
    loader = OntologyLoader('https://raw.githubusercontent.com/isi-vista'
                            '/gaia-interchange/master/src/main/resources'
                            '/edu/isi/gaia/seedling-ontology.ttl')
    loader.as_brat_conf('annotation.conf', 'visual.conf')
    # loader.as_text('temp.txt')
