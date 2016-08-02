#coding:utf-8


class Xml_Parse_base:
    #base class for xml parsing
    #def __init__(self):
    #   a = 0

    def get_attrvalue(self, node, attrname):
        return node.getAttribute(attrname) if node else ''

    def get_nodevalue(self, node, index = 0):
        return node.childNodes[index].nodeValue if node else ''

    def get_xmlnode(self, node, name):
        return node.getElementsByTagName(name) if node else''


