#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:08:02 2021

@author: qichen
"""
from aiida import orm, load_profile
load_profile()
demo_group = orm.Group(label='demo_group2')
demo_group.store()
work_node = orm.load_node('16569')
demo_group.add_nodes(work_node.called)
