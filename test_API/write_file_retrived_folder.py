from aiida.orm import load_node

def write_retrived_files(path,retrived_folder_node):
    retrieved_nodes = load_node(retrieved)
    name_list = retrieved_nodes.list_object_names()
    for i in name_list:
        content = retrieved_nodes.get_object_content(i)
        f = open((path+i), "w")
        f.write(content)
        f.close()

