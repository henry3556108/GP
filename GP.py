from tool.gp_tools import *
from tool.teds import *
import json
import pickle
import os
import time
from typing import List
from datetime import datetime
from tool.tool import copy_json_to_folder

def append_trees_to_csv(trees: List[Node], filename='output.csv', generation:int=0):
    # Try to read existing data from the CSV file, or create an empty DataFrame if the file does not exist
    try:
        df_existing = pd.read_csv(filename)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=['generation', 'file_name', 'fitness_score1', 'fitness_score2', 'test_fitness_score1', 'test_fitness_score2'])

    # Prepare new data for DataFrame
    data = {
        'generation': [],
        'max_depth': [],
        'total_node': [],
        'file_name': [],
        'fitness_score1': [],
        'fitness_score2': [],
        'test_fitness_score1': [],
        'test_fitness_score2': []
    }

    for tree in trees:
        data["max_depth"].append(tree.max_depth())
        data["total_node"].append(tree.total_nodes)
        data["generation"].append(generation)
        data['file_name'].append(tree.file_name)
        data['fitness_score1'].append(tree.fitness_score[0] if len(tree.fitness_score) > 0 else None)
        data['fitness_score2'].append(tree.fitness_score[1] if len(tree.fitness_score) > 1 else None)
        data['test_fitness_score1'].append(tree.test_fitness_score[0] if len(tree.test_fitness_score) > 0 else None)
        data['test_fitness_score2'].append(tree.test_fitness_score[1] if len(tree.test_fitness_score) > 1 else None)

    # Create DataFrame for new data
    df_new = pd.DataFrame(data)
    # Append new data to existing data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates, considering all columns or specific columns
    df_combined = df_combined.drop_duplicates(subset=['file_name'], keep='last')

    # Write the non-duplicate data back to CSV
    df_combined.to_csv(filename, index=False)


def save_to_pending(population: List[Node], dst="pending") -> List[str]:
    file_names = []
    while population:
        individual = population.pop()
        if not individual.finished:
            file_name = os.path.join(dst, individual.file_name)
            with open(file_name, "wb") as f:
                pickle.dump(individual, f)
        file_names.append(individual.file_name)
    return file_names


def clear_useless(population: List[Node], src="finished", dst="fail"):
    pkl_names = os.listdir(src)
    population_names = [individual.file_name for individual in population]
    for pkl_name in pkl_names:
        if pkl_name not in population_names:
            # print(f"in fail {pkl_name}")
            src_file_path = os.path.join(src, pkl_name)
            dst_file_path = os.path.join(dst, pkl_name)
            os.rename(src_file_path, dst_file_path)


def get_population_from_folder(src: str = "finished") -> List[Node]:
    pkl_names = os.listdir(src)
    population = []
    for pkl_name in pkl_names:
        file_path = os.path.join(src, pkl_name)
        with open(file_path, "rb") as f:
            root: Node = pickle.load(f)
            population.append(root)
    return population

def all_finished(population_name_list: List[str], dst="finished"):
    file_names = os.listdir(dst)
    for name in population_name_list:
        if name not in file_names:
            return False
    return True


if __name__ == "__main__":
    config = json.load(open("config.json"))
    generation_times = config["generation"]
    population_size = config["population_size"]
    offsprint_amount = config["offsprint_amount"]
    risk_free_rate = config["risk_free_rate"]
    reborn_rate = config["reborn_rate"]
    depth = config["depth"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_folder = config["record"]
    record_path = os.path.join("record", record_folder)
    record_csv_path = os.path.join(record_path, current_time + ".csv")
    copy_json_to_folder("config.json", record_path)
    population = init_population(population_size)
    population_list = save_to_pending(population)
    for generation in range(1, generation_times + 1):
        time.sleep(1)
        print(generation)
        while not all_finished(population_list):
            print("waitning")
            time.sleep(1)
        population = get_population_from_folder()
        append_trees_to_csv(population, record_csv_path, generation)
        population = filter_trees(population, 0.9)
        
        population = survivor_selection(population, offsprint_amount, int(population_size * (1 - reborn_rate)))
        offsprints: List[Node] = []
        while len(offsprints) < offsprint_amount:
            parent: List[Node] = parent_selection(population)
            # 計算 gp_crossover 的內存使用
            offsprint = gp_crossover(parent[0], parent[1])
            if offsprint:
                offsprint.count_descend()
                offsprints.append(offsprint)
        reborn = []
        if len(offsprints) + len(population) < population_size:
            size = (population_size - (len(offsprints) + len(population))) // 2 * 2
            reborn = init_population(size=size, depth=depth)
            offsprints += reborn
        clear_useless(population)
        population += offsprints
        population_list = [individual.file_name for individual in population]
        save_to_pending(offsprints)

