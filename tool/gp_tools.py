import pandas as pd
import random
from typing import Callable, Tuple, List, Union
import copy
import numpy as np
from tool.basic_class import *
from tool.basic_function import *
import time

op_dic = {(pd.Series, bool): [(bigger, {0: (pd.Series, float), 1: (pd.Series, float)}), 
                              (crossover, {0: (pd.Series, float), 1: (pd.Series, float)})],
          (pd.Series, float): [(moving_average, {0: (pd.Series, float), 1: (int,)}), 
                               (ema, {0: (pd.Series, float), 1: (int,)}),
                               (multi, {0: (pd.Series, float), 1: (int,)}),
                               (add, {0: (pd.Series, float), 1: (pd.Series, float)}),
                               (macd, {0: (pd.Series, float), 1: (int,), 2: (int,), 3: (int,)}), 
                               (zscore, {0: (pd.Series, float), 1: (int,)}),
                               (ts_max, {0: (pd.Series, float), 1: (int,)}),
                               (ts_min, {0: (pd.Series, float), 1: (int,)}),
                               (shift, {0: (pd.Series, float), 1: (int,)}), 
                               (count, {0: (pd.Series, float), 1: (int,)}), 
                               (k_value, {0: (pd.DataFrame, None), 1: (int,)}), 
                               (d_value, {0: (pd.DataFrame, None), 1: (int,)}), 
                               (rsi, {0: (pd.Series, float), 1: (int,)})]
          }

# TODO 資料的初始化這邊還需要實作，目前是以最簡單的資料及去做出來的
df = pd.read_csv("dataset/SPY current.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)


df_bull1 = pd.read_csv("dataset/SPY bull 1.csv")
df_bull1["Date"] = pd.to_datetime(df_bull1["Date"])
df_bull1["0"] = 0
df_bull1["1"] = 1

df_bull1.set_index("Date", inplace=True)

df_bear1 = pd.read_csv("dataset/SPY bear 1.csv")
df_bear1["Date"] = pd.to_datetime(df_bear1["Date"])
df_bear1["0"] = 0
df_bear1["1"] = 1

df_bear1.set_index("Date", inplace=True)

df_bull2 = pd.read_csv("dataset/SPY bull 2.csv")
df_bull2["Date"] = pd.to_datetime(df_bull2["Date"])
df_bull2["0"] = 0
df_bull2["1"] = 1
df_bull2.set_index("Date", inplace=True)

df_bear2 = pd.read_csv("dataset/SPY bear 2.csv")
df_bear2["Date"] = pd.to_datetime(df_bear2["Date"])
df_bear2["0"] = 0
df_bear2["1"] = 1
df_bear2.set_index("Date", inplace=True)

# 用舊的資料當作訓練
train_datasets = [df_bull1, df_bear1]

# 新的資料及當作驗證
test_datasets = [df_bull2, df_bear2]
data_dic = {(pd.Series, float): ["Close", "Open", "High", "Low", "0", "1"],
            (pd.DataFrame, None): ["Whole"],
            (int,): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 21, 26, 60, 63, 120, 125, 240, 250]}

def generate_node(key, depth) -> Node:
    if depth == 0 or key not in op_dic:
        return Node(random.choice(data_dic[key]), key)
    
    func, params_dict = random.choice(op_dic[key])
    node = Node(func, key)
    
    for param_index, param_key in params_dict.items():
        child = generate_node(param_key, depth - 1)
        node.parameters.append(child)
        child.father = node
    
    return node

# 随机生成树
def generate_tree(depth: int) -> Node:
    root_key = (pd.Series, bool)
    return generate_node(root_key, depth)



def init_population(size: int = 200, depth: int = 4) -> List[Node]:
    '''
    預設 depth 為 4 是因為採用 50-50 策略
    '''
    # TODO 這邊要做 root 結構 Tree，要有左右子樹，左邊子樹為 buy 右邊為 sell
    population: List[Node] = []
    for _ in range(size//2):
        root = Node(None, None)
        buy_side, sell_side = generate_tree(depth), generate_tree(depth)
        buy_side.father = sell_side.father = root
        root.parameters.append(buy_side)
        root.parameters.append(sell_side)
        population.append(root)
    for _ in range(size//2):
        root = Node(None, None)
        buy_side, sell_side = generate_tree(depth//2), generate_tree(depth//2)
        buy_side.father = sell_side.father = root
        root.parameters.append(buy_side)
        root.parameters.append(sell_side)
        population.append(root)    
    for individual in population:
        individual.file_name = str(hash(time.time())) + ".pkl"
        individual.count_descend()
    return population


def collect_nodes(root):
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.extend(node.parameters)
        stack.extend(node.parameters)
    return nodes

# def collect_nodes(node: Node):
#     """
#     蒐集該 node 中不為 leaf 的 node
#     """
#     non_leaf_nodes = []

#     def _collect(node, is_root=False):
#         if not is_root:  # Check if the node has children and is not the root
#             non_leaf_nodes.append(node)
#         for child in node.parameters:
#             _collect(child)

#     _collect(node, True)
#     return non_leaf_nodes


def select_node(root, random_choice=True, constrant=None) -> Node:
    """
    根據該 root 裡面的 buy side 跟 sell side 所有非 leaf node 隨機挑選一個
    """
    nodes = []
    for child in root.parameters:
        nodes += collect_nodes(child)
    if random_choice:
        return random.choice(nodes)
    else:
        constranted_nodes = []
        for node in nodes:
            if node.key == constrant:
                constranted_nodes.append(node.father)
        return random.choice(constranted_nodes)


def gp_crossover(p1: Node, p2: Node) -> Node:
    # 如果下面這一行失敗代表，node1 所需要 crossover 的東西 p2 之中沒有，在越淺層越有可能有這種事情發生
    try:
        offspring = copy.deepcopy(p2)
        new_p1 = copy.deepcopy(p1)
        node1 = select_node(new_p1)
        key = node1.key
        node2 = select_node(offspring, random_choice=False, constrant=key)
        for child_pos, child in enumerate(node2.parameters):
            # TODO 這邊的挑選一定會優先挑選到第一個 參數 需要修正
            if child.key == key:
                node2.parameters[child_pos] = node1
                node1.father = node2
                break
        offspring.fitness_score = []
        offspring.test_fitness_score = []
        offspring.finished = False
        offspring.file_name = str(hash(time.time())) + ".pkl"
        return offspring
    except Exception as e:
        return None

def calculate_total_transaction_fees(transaction_amount: float, shares_sold=1):
    '''
    transaction_amount: 交易金額 理應為 close
    shares_sold: 賣出股票數量
    '''
    # 證監會收費（僅對賣單收取）
    sec_fee = max(0.00000008 * transaction_amount, 0.01)

    # 交易活動費（僅對賣單收取）
    finra_fee = 0.000166 * shares_sold
    finra_fee = min(max(finra_fee, 0.01), 8.3)

    # 交收費
    clearing_fee = 0.003 * shares_sold
    max_clearing_fee = 0.03 * transaction_amount
    clearing_fee = min(max(clearing_fee, 0.01), max_clearing_fee)

    # 計算總交易費用
    total_fees = sec_fee + finra_fee + clearing_fee
    return total_fees


def calculate_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

def calculate_total_return(prices):
    prices = prices.to_list()
    total_return = (prices[-1] / prices[0]) - 1
    return total_return

def simulate(buy_side: Node, sell_side: Node, close, risk_free_rate=0):
    # 取得 buy_signal 和 sell_signal
    buy_signal = buy_side.get_result()
    buy_signal.name = "buy_signal"
    sell_signal = sell_side.get_result()
    sell_signal.name = "sell_signal"
    
    # 設置 close 欄位名稱
    close.name = "Close"
    # 合併成一個 DataFrame
    df = pd.concat([sell_signal, buy_signal, close], axis=1)

    # 未來這一段要改成 shift open
    df["next_close"] = df["Close"].shift(1)
    df.reset_index(inplace=True)
    total_days = len(df)
    hold = False
    entry_index = None
    transactions = []
    infos = {}
    
    for index, row in df.iterrows():
        if hold == False and row["buy_signal"] == True:
            entry_index = index
            hold = True
        elif hold == True and row["sell_signal"] == True:
            hold = False
            transactions.append((entry_index, index))
    if hold:
        transactions.append((entry_index, index))
        
    df['Returns'] = df['Close'].pct_change()
    
    for start_date, end_date in transactions:
        holding_returns_pct = df.loc[start_date:end_date, 'Returns']
        holding_close = df.loc[start_date:end_date, 'Close']
        sharpe_ratio = calculate_sharpe_ratio(holding_returns_pct.dropna(), risk_free_rate)
        total_return = calculate_total_return(holding_close)
        infos[(start_date, end_date)] = {"sharpe_ratio": sharpe_ratio, "total_return": total_return, "holding_days": end_date - start_date}
    return infos, total_days, df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义计算fitness score的函数
def calculate_fitness_score(infos: dict, target_sharpe=0.8, target_cagr=0.05, risk_free_rate = 0.05):
    # Calculate the average sharpe ratio and average total return
    total_sharpe_ratio = sum([entry['sharpe_ratio'] / (1 + np.exp(-entry["holding_days"])) for entry in infos.values()])
    total_total_return = sum([entry['total_return'] / (1 + np.exp(-entry["holding_days"])) for entry in infos.values()])
    total_days = len(infos)
    average_sharpe_ratio = total_sharpe_ratio / total_days
    average_total_return = total_total_return / total_days
    if average_sharpe_ratio > target_sharpe and average_total_return > target_cagr:
        return (1 + average_sharpe_ratio - target_sharpe) * (1 + (average_total_return - target_cagr)) * (sigmoid(len(infos)) + 0.12) ** 2 # 如果交易次數少於 2 次，計算出來的結果會小於 1 會有懲罰效果
    else:
        return 0
    
def tournament_selection(population: List[Node], tournament_size: int) -> Node:
    # 隨機選取 tournament_size 個個體
    tournament_contestants = random.sample(population, tournament_size)
    # 找出 fitness_score 最高的個體
    winner = max(tournament_contestants, key=lambda node: sum(node.fitness_score))
    return winner


def survivor_selection(population: List[Node], offsprint_amount: int, population_size: int) -> List[Node]:
    last_index = len(population) - (population_size - offsprint_amount)
    # 過濾掉 fitness_score 為 np.nan 的個體
    filtered_population = [
        node for node in population if not len(node.fitness_score) == 0]
    # 將 population 按照 fitness_score 進行排序，從高到低
    sorted_population = sorted(
        filtered_population, key=lambda node: sum(node.fitness_score), reverse=True)
    # 保留前 len(sorted_population) - offsprint_amount 個個體
    for _ in range(last_index):
        sorted_population.pop()
    return sorted_population

def parent_selection(population: List[Node]):
    parent = []
    while len(parent) < 2:
        individual = tournament_selection(population, 2)
        parent.append(individual)
    return parent

def top_5_percent_mean(population: List[Node]) -> float:
    # 將 population 按照 fitness_score 進行排序，從高到低
    sorted_population = sorted(
        population, key=lambda node: sum(node.fitness_score), reverse=True)
    # 取出前 5% 的個體
    top_5_percent_count = max(1, len(sorted_population) * 5 // 100)  # 至少保留一個個體
    top_5_percent_population = sorted_population[:top_5_percent_count]
    # 計算前 5% 個體的 fitness_score 平均值
    mean_fitness_score = np.mean(
        [sum(node.fitness_score) / len((node.fitness_score)) for node in top_5_percent_population])
    return mean_fitness_score

def filter_inf_and_average(data:List[np.array]) -> float:
    # 过滤掉 -np.inf 并计算每个样本的平均值
    filtered_averages = []
    for sample in data:
        filtered_sample = sample[sample != -np.inf]
        if filtered_sample.size > 0:
            filtered_averages.append(np.mean(filtered_sample))# print(filtered_averages)
    # # 将平均值排序并找出前 5% 的数据点
    top_5_percent_threshold = np.percentile(filtered_averages, 95)
    top_5_percent = [value for value in filtered_averages if value >= top_5_percent_threshold]
    return np.mean(top_5_percent)

# if __name__ == "__main__":
#     # 建立一個示範 population
#     population = [random.uniform(0, 100) for _ in range(10)]
#     for i in range(len(population)):
#         node = Node(None, None)
#         node.fitness_score = population[i]
#         population[i] = node
#     for i, individual in enumerate(population):
#         print(f"Individual {i}: Fitness Score = {individual.fitness_score}")

#     tournament_size = 3  # 可以根據需要調整 tournament_size

#     winner = tournament_selection(population, tournament_size)
#     print(f"Winner Fitness Score: {winner.fitness_score}")

# if __name__=="__main__":
#     # 生成一個深度為 3，每層最多 3 個子節點的樹
#     # population = init_population(2)
#     p1, p2 = init_population(4)
#     graph = visualize_tree(p2)
#     graph.render('before output_tree2')
#     offspring = crossover(p1, p2)
#     graph = visualize_tree(p1)
#     graph.render('output_tree1')
#     graph = visualize_tree(p2)
#     graph.render('after output_tree2')
    # graph = visualize_tree(offspring)
    # graph.render('output_tree3')
    # print(p1)
    # print(p2)
    # print(offspring)
    # for individual in population:
    #     print(individual.get_result())
    # random_tree = generate_tree(2)
    # result = random_tree.get_result()
    # print(result)
    # graph = visualize_tree(random_tree)
    # graph.render('output_tree')
