from tool.gp_tools import *
from tool.teds import *
from tool.monte_carlo import *
import pickle
import os
from datetime import datetime
from GP import get_population_from_folder
import time
import json

if __name__=="__main__":
    pending_folder = "pending"
    finished_folder = "finished"
    config = json.load(open("config.json"))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_folder = os.path.join("record", config["record"])
    target_stragety = get_target_stragety()
    
    while True:
        time.sleep(3)
        population: List[Node] = get_population_from_folder(pending_folder)
        fitness_scores = []
        start_time = time.time()
        for individual in population:
            if not individual.finished:
                fitness_score = calculate_mo_fitness_score(individual, target_stragety)
                individual.fitness_score.append(fitness_score)
                print(individual.fitness_score)
                fitness_scores.append(fitness_score)
        spend_time = time.time() - start_time
        for individual in population:
            individual.finished = True
            file_path = os.path.join(finished_folder, individual.file_name)
            with open(file_path, 'wb') as file:
                pickle.dump(individual, file)
            old_file = os.path.join(pending_folder, individual.file_name)
            print(individual.fitness_score)
            os.remove(old_file)
        if len(population) != 0:
            fitness_scores = [fitness_score for fitness_score in np.array(fitness_scores) if not np.isnan(fitness_score)]
            with open(os.path.join(record_folder, current_time + ".txt"), 'a') as f:
                line = ','.join(map(str, fitness_scores))  # 將數字轉換為逗號分隔的字符串
                f.write(line + " " + str(spend_time) + '\n')