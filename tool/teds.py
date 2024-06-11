from tool.gp_tools import Node
from typing import List, Tuple, Callable
import numpy as np
import random

def min_distance_matching(params1: List[Node], params2: List[Node]) -> List[int]:
    distances = []
    for i in range(len(params1)):
        for j in range(len(params2)):
            distances.append(tree_edit_distance(params1[i], params2[j]))
    return sorted(distances)[:abs(len(params1) - len(params2))]

def tree_edit_distance(tree1: Node, tree2: Node) -> int:
    if tree1 is None and tree2 is None:
        return 0
    if tree1 is None:
        return tree2.total_nodes
    if tree2 is None:
        return tree1.total_nodes

    m = len(tree1.parameters)
    n = len(tree2.parameters)
    distance_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        distance_matrix[i][0] = tree1.parameters[i - 1].total_nodes  # 刪除成本
    for j in range(1, n + 1):
        distance_matrix[0][j] = tree2.parameters[j - 1].total_nodes  # 插入成本

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = tree_edit_distance(tree1.parameters[i - 1], tree2.parameters[j - 1])
            distance_matrix[i][j] = min(distance_matrix[i - 1][j] + tree1.parameters[i - 1].total_nodes,       # 刪除
                                        distance_matrix[i][j - 1] + tree2.parameters[j - 1].total_nodes,       # 插入
                                        distance_matrix[i - 1][j - 1] + cost) # 替換

    node_diff = abs(len(tree1.parameters) - len(tree2.parameters))
    if m != n:
        unmatched_distance = sum(min_distance_matching(tree1.parameters, tree2.parameters))
        node_diff += unmatched_distance
    
    if isinstance(tree1.data, Callable) and isinstance(tree2.data, Callable):
        # print(distance_matrix[m][n], tree1.data, tree2.data , node_diff)
        return distance_matrix[m][n] + (1 if tree1.data != tree2.data else 0) + node_diff
    else:
        return distance_matrix[m][n] + node_diff
    
def calculate_similarity(tree1: Node, tree2: Node) -> float:
    edit_dist = tree_edit_distance(tree1, tree2)
    max_nodes = tree1.total_nodes if tree1.total_nodes > tree2.total_nodes else tree2.total_nodes
    similarity = max(0, 1 - edit_dist / max_nodes)  # 保證相似度在 0 到 1 之間
    return similarity

def build_similarity_matrix(trees: List[Node], threshold: float = 0.9) -> np.ndarray:
    n = len(trees)
    similarities = np.zeros((n, n))
    
    # 計算所有樹之間的相似度矩陣
    for i in range(n):
        for j in range(i + 1, n):
            similarities[i][j] = 1 if calculate_similarity(trees[i], trees[j]) > threshold else 0
            similarities[j][i] = similarities[i][j]
    
    return similarities

def filter_trees(trees: List[Node], threshold: float = 0.95, iterations: int = 2) -> List[Node]:
    n = len(trees)
    similarities = build_similarity_matrix(trees, threshold)
    # print(similarities)
    # 優先保留 fitness_score 最高的樹
    # 下面這一行是實驗一的
    best_tree_index = max(range(n), key=lambda i: sum(trees[i].fitness_score))
    # 下面這一行是實驗二的
    # best_tree_index = max(range(n), key=lambda i: trees[i].fitness_score[0])
    print(best_tree_index, trees[best_tree_index].fitness_score[0])
    best_set = set()
    max_size = 0
    
    for _ in range(iterations):
        current_set = {best_tree_index}
        remaining_indices = set(range(n)) - {best_tree_index}
        
        while remaining_indices:
            index = random.choice(tuple(remaining_indices))
            if all(similarities[index][j] <= threshold for j in current_set):
                current_set.add(index)
            remaining_indices.discard(index)
        
        if len(current_set) > max_size:
            max_size = len(current_set)
            best_set = current_set
    del similarities
    return [trees[i] for i in best_set]
