#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:28:17 2021

@author: qichen
"""
from aiida import load_profile
profile = load_profile()
from aiida.common import LinkType
from aiida.orm.utils.links import LinkPair
from aiida.tools.visualization import Graph, pstate_node_styles

graph = Graph()
graph.add_node('15870')
graph.add_node('15879')
graph.add_node('15888')
graph.add_node('15898')
graph.add_node('15908')
graph.add_node('15917')

graph.add_incoming('15870')
graph.add_outgoing('15870')

graph.add_incoming('15888')
graph.add_outgoing('15888')

graph.add_outgoing('15879')

graph.add_outgoing('15888')

graph.add_outgoing('15898')

graph.add_outgoing('15908')

graph.add_outgoing('15917')



graph.graphviz