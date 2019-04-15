"""
Read NeGra format.
"""

import xml.etree.ElementTree as ET


class TreeNode:
    def __init__(self, node_id, is_root=False):
        self.node_id = node_id
        self.children = []
        self.attributes = {}
        self.is_root = is_root

    def add_child(self, node):
        self.children.append(node)

    def add_attribute(self, attribute_name, attribute_value):
        self.attributes[attribute_name] = attribute_value


class NeGraXML:
    def __init__(self):
        self.c_parses = []
        self.f_parses = []

    def load_data(self, xml_file):
        root = ET.parse(xml_file).getroot()
        body = root.find("body")

        for sent in body:
            sent_id = sent.tag['id']
            c_graph_node = sent.find("graph")
            sem_node = sent.find("sem")

            self.add_constituent_parse(c_graph_node)
            self.add_frame_parse(sem_node)

    def add_frame_parse(self, sem_node):
        pass

    def add_constituent_parse(self, c_graph_node):
        root_id = c_graph_node.attrib['root']

        term_nodes = c_graph_node.find('terminals')
        nt_nodes = c_graph_node.find('nonterminals')

        token_nodes = []

        id2node = {}

        for t_node in term_nodes:
            attrib = t_node.attrib
            token_node = TreeNode(attrib['id'])
            token_node.add_attribute('tag', attrib['cat'])
            token_nodes.append(token_node)
            id2node[token_node.node_id] = token_node

        for nt_node in nt_nodes:
            attrib = nt_node.attrib
            nonterm_node = TreeNode(attrib['id'], root_id == attrib['id'])
            nonterm_node.add_attribute('tag', attrib['cat'])

            for edge_node in nt_node.findall('edge'):
                child_id = edge_node['idref']
                child_node = id2node[child_id]
                nonterm_node.add_child(child_node)
            id2node[nonterm_node.node_id] = nonterm_node

        self.c_parses.append(
            {
                'tokens': token_nodes,
                'id2node': id2node,
            }
        )


if __name__ == '__main__':
    import sys

    negra_xml = sys.argv[1]

    negra_parser = NeGraXML()
    negra_parser.load_data(negra_xml)
