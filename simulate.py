from tool.gp_tools import *
from tool.teds import *
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
    
    
    while True:
        time.sleep(3)
        population: List[Node] = get_population_from_folder(pending_folder)
        cagrs = []
        for train_dataset in train_datasets:
            start_time = time.time()
            tmp_df = train_dataset.copy()
            tmp_df['Returns'] = tmp_df['Close'].pct_change()
            buy_and_hold_return = calculate_total_return(tmp_df['Close'])
            buy_and_hold_sharpe_ratio = calculate_total_return(tmp_df['Returns'].dropna())
            for individual in population:
                if not individual.finished:
                    individual.set_source_data(train_dataset)
                    infos, total_days, _ = simulate(individual.parameters[0], individual.parameters[1], train_dataset["Close"])
                    if len(infos) == 0:
                        fitness_score = -np.inf
                        cagrs.append(np.nan)
                    else:
                        fitness_score = calculate_fitness_score(infos, target_sharpe=buy_and_hold_sharpe_ratio, target_cagr=buy_and_hold_return, risk_free_rate=0.05)
                        cagrs.append(np.array([entry["total_return"] for entry in infos.values()]).mean())
                    individual.fitness_score.append(fitness_score)
        for test_dataset in test_datasets:
            tmp_df = test_dataset.copy()
            buy_and_hold_return = calculate_total_return(tmp_df['Close'])
            for individual in population:
                if not individual.finished:
                    individual.set_source_data(test_dataset)
                    infos, total_days, _ = simulate(individual.parameters[0], individual.parameters[1], test_dataset["Close"])
                    if len(infos) == 0:
                        fitness_score = -np.inf
                        cagrs.append(np.nan)
                    else:
                        fitness_score = calculate_fitness_score(infos, target_sharpe=buy_and_hold_sharpe_ratio, target_cagr=buy_and_hold_return, risk_free_rate=0.05)
                        cagrs.append(np.array([entry["total_return"] for entry in infos.values()]).mean())
                    individual.test_fitness_score.append(fitness_score)            
        spend_time = time.time() - start_time
        for individual in population:
            individual.finished = True
            file_path = os.path.join(finished_folder, individual.file_name)
            with open(file_path, 'wb') as file:
                pickle.dump(individual, file)
            old_file = os.path.join(pending_folder, individual.file_name)
            os.remove(old_file)
        if len(population) != 0:
            filtered_cagrs = [cagr for cagr in np.array(cagrs) if not np.isnan(cagr)]
            with open(os.path.join(record_folder, current_time + ".txt"), 'a') as f:
                line = ','.join(map(str, filtered_cagrs))  # 將數字轉換為逗號分隔的字符串
                
                f.write(line + " " + str(spend_time) + '\n')