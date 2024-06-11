from tool.gp_tools import *
from tool.teds import *
import json
import pickle
import os
from datetime import datetime
from matplotlib import pyplot as plt
import gc
from tool.tool import analyze_memory_usage
import time



def records(start_time, *args, **kargs):      
    # 確認 record 資料夾是否存在，若不存在則創建
    if not os.path.exists("record"):
        os.makedirs("record")
    
    
    # 檔案路徑
    file_path = os.path.join("record", f"record_{start_time}.txt")
      
    with open(file_path, "a") as f:
        for arg in args:
            f.write(f"{arg}\n")
        for key, value in kargs.items():
            f.write(f"{key}: {value}\n")

def plot_trading_fig(data, file_name):
    # 假設 CSV 文件包含 'close', 'buy_signal', 'sell_signal' 列
    close = data['Close']
    buy_signal = data['buy_signal']
    sell_signal = data['sell_signal']

    # 進出場邏輯
    holding = False
    buy_signals = []
    sell_signals = []

    for i in range(len(close)):
        if not holding and buy_signal[i]:
            buy_signals.append((i, close[i]))
            holding = True
        elif holding and sell_signal[i]:
            sell_signals.append((i, close[i]))
            holding = False
    # 如果持倉到最後一天都還沒賣出就直接 sell
    if holding:
        sell_signals.append((i, close[i]))

    # 視覺化
    plt.figure(figsize=(14, 7))
    plt.plot(close, label='Close Price', color='blue')
    for buy in buy_signals:
        plt.annotate('Buy', xy=buy, xytext=(buy[0], buy[1]+1),
                    arrowprops=dict(facecolor='green', shrink=0.05))
    for sell in sell_signals:
        plt.annotate('Sell', xy=sell, xytext=(sell[0], sell[1]+1),
                    arrowprops=dict(facecolor='red', shrink=0.05))

    plt.title('Close Price with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name+".png")
def get_memory_usage(snapshot1, snapshot2):
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_size = sum(stat.size for stat in stats)
    return total_size


if __name__ == "__main__":
    config = json.load(open("config.json"))
    generation_times = config["generation"]
    population_size = config["population_size"]
    offsprint_amount = config["offsprint_amount"]
    risk_free_rate = config["risk_free_rate"]
    depth = config["depth"]
    # 開始跟踪內存分配
    # tracemalloc.start()

    population = init_population(size = population_size, depth = depth)
    # 獲取當前日期與時間，格式化為字符串
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        for generation in range(1, generation_times + 1):
            cagrs = []
            print(f"Generation:{generation}, population size:{len(population)}")
            start_time = time.time()
            # 計算訓練資料集的 fitness score
            for train_dataset in train_datasets:
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
                    
            print(f"generation {generation} simulation cost:{time.time() - start_time}")
            start_time = time.time()
            testing_fitness_scores = []
            training_fitness_scores = []
            # 計算驗證資料集的 fitness_score
            for individual in population:
                testing_fitness_scores.append(np.array(individual.test_fitness_score))
                training_fitness_scores.append(np.array(individual.fitness_score))
                individual.finished = True
            train_score = filter_inf_and_average(training_fitness_scores)
            test_score = filter_inf_and_average(testing_fitness_scores)
            print("----------")
            print(f"training_fitness_scores is: {train_score}")
            print(f"testing_fitness_scores is: {test_score}")
            print("----------")
            filtered_cagrs = [cagr for cagr in np.array(cagrs) if not np.isnan(cagr)]
            
            # survivor selection
            population = survivor_selection(population, offsprint_amount)
            print(f"generation {generation} selection cost:{time.time() - start_time}")
            start_time = time.time()

            # filter 太高相似性的 tree
            population = filter_trees(population)
            print(f"generation {generation} filter_trees cost:{time.time() - start_time} left amount:{len(population)}")
            each_tree_node_amount = [individual.total_nodes for individual in population]
            each_tree_max_depth = [individual.max_depth() for individual in population]
            max_node = max(each_tree_node_amount)
            max_depth = max(each_tree_max_depth)
            print(f"generation {generation} top 5% fitness_score is:{top_5_percent_mean(population)}, max_node:{max_node}, max depth:{max_depth}")            
            # while population:
            #     individual = population.pop()
            #     file_name = str(hash(individual)) + ".pkl"
            #     with open(file_name, 'wb') as file:
            #         individual.file_name = file_name
            #         pickle.dump(individual, file)

            if generation == generation_times:
                break
            offsprints = []
            count = 0
            crossover_times = 0
            start_time = time.time()
            while len(offsprints)< offsprint_amount:
                parent: List[Node] = parent_selection(population)                
                # 計算 gp_crossover 的內存使用
                offsprint = gp_crossover(parent[0], parent[1])
                if offsprint:
                    crossover_times += 1
                    # 計算 count_descend 的內存使用
                    offsprint.count_descend()
                    offsprints.append(offsprint)
                else:
                    count += 1
            # 手动运行垃圾回收器
            gc.collect()
            print(f"generation {generation} generate offspring cost:{time.time() - start_time} total fail:{count} crossover time:{crossover_times}")
            start_time = time.time()
            reborn = []
            if len(offsprints) + len(population) < population_size:
                size = (population_size - (len(offsprints) + len(population))) // 2 * 2
                reborn = init_population(size = size, depth = depth)
                offsprints += reborn
            print(f"generation {generation} reborn cost:{time.time() - start_time} reborn amount:{len(reborn)}")
            records(current_time, generation, top_5_percent_mean=top_5_percent_mean(population), mean_cagr=np.array(filtered_cagrs).mean())
            population += offsprints
            print(f"end generation {generation}")
    finally:
        for index, individual in enumerate(population):
            if individual.fitness_score != None:
                if not os.path.exists(f"stragety/{current_time}"):
                    os.makedirs(f"stragety/{current_time}")
                file_name = f'stragety/{current_time}/indivudual{index}_fitness_score{individual.fitness_score}'
                individual.file_name = file_name
                with open(file_name + '.pkl', 'wb') as file:
                    pickle.dump(individual, file)
        sorted_population = sorted(population, key=lambda node: sum(node.fitness_score), reverse=True)
        # 取出前 5% 的個體
        top_5_percent_count = max(1, len(sorted_population) * 5 // 100)  # 至少保留一個個體
        top_5_percent_population = sorted_population[:top_5_percent_count]
        print(len(top_5_percent_population))
        for individual in top_5_percent_population:
            try:
                for index, dataset in enumerate(train_datasets):
                    tmp_df = dataset.copy()
                    buy_and_hold_return = calculate_total_return(tmp_df['Close'])
                    individual.set_source_data(dataset)
                    infos, total_days, trading_record = simulate(individual.parameters[0], individual.parameters[1], dataset["Close"])
                    if len(infos) != 0:
                        file_name = individual.file_name+f"_{index}_org"
                        plot_trading_fig(trading_record, file_name)
                individual.set_source_data(df)
                infos, total_days, trading_record = simulate(individual.parameters[0], individual.parameters[1], df["Close"])
                print(individual.file_name, len(infos))
                if len(infos) != 0:
                    file_name = individual.file_name
                    plot_trading_fig(trading_record, file_name)
            except Exception as e:
                print(e)
                pass