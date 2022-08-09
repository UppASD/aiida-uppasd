# -*- coding: utf-8 -*-
"""
Auxiliary function to write the retrieved files to a local path
"""

from aiida.orm import load_node


def write_retrieved_files(path, retrieved_folder_node):
    """
    Write the retrieved files to disk

    :param path: path to the folder where data will be written
    :type path: str
    :param retrieved_folder_node: identifier for the node
    :type retrieved_folder_node: str
    """
    retrieved_nodes = load_node(retrieved_folder_node)
    name_list = retrieved_nodes.list_object_names()
    for _name in name_list:
        content = retrieved_nodes.get_object_content(_name)
        with open((path + _name), 'w') as handler:
            handler.write(content)
