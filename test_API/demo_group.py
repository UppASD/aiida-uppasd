#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:08:02 2021

@author: qichen
"""
from aiida.orm import Group, load_node
import aiida
aiida.load_profile()
demo_group = Group(label='demo_group2')
demo_group.store()
work_node = load_node('16569')
demo_group.add_nodes(work_node.called)
