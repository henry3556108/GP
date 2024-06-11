from tool.basic_class import Node
import gc
import sys
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
from tool.gp_tools import *
import os
import shutil

def copy_json_to_folder(source_file_path: str, target_folder: str):
    # 確保目標文件夾存在，如果不存在則創建
    os.makedirs(target_folder, exist_ok=True)

    # 構建目標文件的完整路徑
    file_name = os.path.basename(source_file_path)
    target_file_path = os.path.join(target_folder, file_name)

    # 複製文件
    shutil.copy(source_file_path, target_file_path)
    print(f"File copied to {target_file_path}")

def add_nodes_edges(dot, node, parent_id=None):
    if isinstance(node.data, str):
        label = f"Function: {node.data}\nKey: {node.key}\nFitness: {node.fitness_score}"
    elif isinstance(node.data, pd.Series):
        label = f"Series: {node.data.name}\nKey: {node.key}\nFitness: {node.fitness_score}"
    else:
        label = f"Data: {node.data}\nKey: {node.key}\nFitness: {node.fitness_score}"
    
    node_id = str(id(node))
    dot.node(node_id, label)
    
    if parent_id:
        dot.edge(parent_id, node_id)
    
    for param in node.parameters:
        add_nodes_edges(dot, param, node_id)

def visualize_tree(root: Node, filename: str):
    dot = Digraph(comment='Tree Visualization')
    add_nodes_edges(dot, root)
    dot.render(filename, format='png', cleanup=True)

def delete_subtree(node: Node):    
    # 遞迴刪除子節點
    while node.parameters:
        child = node.parameters.pop()
        delete_subtree(child)
    # 最後刪除當前節點
    del node


# 计算对象及其子对象的总大小
def get_total_size(obj: Any, seen=None) -> int:
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # 处理对象的所有引用
    referents = gc.get_referents(obj)
    for ref in referents:
        size += get_total_size(ref, seen)

    return size

# 分析内存使用情况，按类型分类
def analyze_memory_usage(objects: List[Any]) -> Dict[type, int]:
    memory_usage = {}
    for obj in objects:
        obj_type = type(obj)
        try:
            obj_size = get_total_size(obj)
            if obj_type not in memory_usage:
                memory_usage[obj_type] = 0
            memory_usage[obj_type] += obj_size
            print(memory_usage)
        except Exception as e:
            print(f"Error processing object of type {obj_type}: {e}")
    return memory_usage



if __name__ == "__main__":
    # 随机生成树节点

    visualize_tree(generate_tree(5), "test")