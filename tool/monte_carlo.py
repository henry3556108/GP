import pandas as pd
import numpy as np
from tool.gp_tools import *
from tool.basic_function import *
import random

def calculate_mo_fitness_score(strategy1, strategy2, num_samples=100):
    score_accumulator = []
    for _ in range(num_samples):
        dataset = generate_stock_data(trend=random.choice(["bullish", "bearish", "neutral"]), amplitude=2, cycle_period=random.randint(20, 120))
        strategy1.set_source_data(dataset)
        strategy2.set_source_data(dataset)
        infos1, total_days1, _ = simulate(strategy1.parameters[0], strategy1.parameters[0], dataset["Close"])
        infos2, total_days2, _ = simulate(strategy2.parameters[0], strategy2.parameters[0], dataset["Close"])
        keys1 = list(infos1.keys())
        keys2 = list(infos2.keys())
        min_keys = min(len(keys1), len(keys2))
        max_keys = max(len(keys1), len(keys2))
        if min_keys == 0 or keys1[0][0] == keys1[0][1]:
            return 0
        key1, key2 = random.choice(list(zip(keys1, keys2)))

        data1 = infos1[key1]
        data2 = infos2[key2]
        # 计算两个策略在三个指标上的欧氏距离
        distance = np.sqrt((data1['sharpe_ratio'] - data2['sharpe_ratio'])**2 +
                            (data1['total_return'] - data2['total_return'])**2 +
                            (data1['holding_days'] - data2['holding_days'])**2)
        score_accumulator.append(distance)
        # 返回平均距离的倒数作为相似度得分，相似度越高，得分越高

    mean_distance = np.mean([score for score in score_accumulator if not np.isnan(score)])
    if mean_distance == 0:
        return float('inf')  # 如果平均距离为零，返回无穷大
    else:
        return 1 / mean_distance  # 使用距离的倒数作为得分

def generate_stock_data(days=252, trend='neutral', volatility=0.01, cycle_period=60, amplitude=0.5):
    # 生成股票資料
    if trend == 'bullish':
        mu = 0.005  # 牛市，价格总体上升
    elif trend == 'bearish':
        mu = -0.005  # 熊市，价格总体下降
    else:
        mu = 0  # 中性市场

    # 隨機生成訊號
    random_walk = np.cumsum(np.random.normal(mu, volatility, days))
    
    # 增加周期性（使用 sin）
    # cycle_period 控制周期長度，amplitude 控制振幅大小
    cyclical_component = amplitude * np.sin(np.linspace(0, 2 * np.pi * days / cycle_period, days))
    prices = 100 * np.exp(random_walk + cyclical_component)  # 通过 e^x 来确保价格始终为正

    high = prices * (1 + np.random.normal(0, 0.02, days))
    low = prices * (1 - np.random.normal(0, 0.02, days))
    close = prices + np.random.normal(0, 2, days)
    volume = np.random.randint(10000, 1000000, days)
    
    # 创建 DataFrame
    stock_data = pd.DataFrame({
        'Open': prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })

    stock_data["0"] = 0
    stock_data["1"] = 1
    return stock_data

def get_target_stragety():
    node_close = Node("Close", (pd.Series, float))
    node_12 = Node(12, (int,))
    node_26 = Node(26, (int,))
    node_9 = Node(9, (int,))

    # 创建 MACD 节点并连接其子节点
    node_macd_buy = Node(macd, (pd.Series, float))
    node_macd_buy.add_child(node_close)
    node_macd_buy.add_child(node_12)
    node_macd_buy.add_child(node_26)
    node_macd_buy.add_child(node_9)

    node_macd_sell = Node(macd, (pd.Series, float))
    node_macd_sell.add_child(node_close)
    node_macd_sell.add_child(node_12)
    node_macd_sell.add_child(node_26)
    node_macd_sell.add_child(node_9)

    # 创建 Cross Over 节点并连接其子节点
    node_cross_over_buy = Node(crossover, (pd.Series, bool))
    zero = Node("0", (pd.Series, float))
    node_cross_over_buy.add_child(node_macd_buy)
    node_cross_over_buy.add_child(zero)

    node_cross_over_sell = Node(crossover, (pd.Series, bool))
    node_cross_over_sell.add_child(zero)
    node_cross_over_sell.add_child(node_macd_sell)


    # 创建根节点并连接其子节点
    root = Node(None, None)
    root.add_child(node_cross_over_buy)
    root.add_child(node_cross_over_sell)
    return root

if __name__=="__main__":
    # 创建底层节点
    node_close = Node("Close", (pd.Series, float))
    node_12 = Node(12, (int,))
    node_26 = Node(26, (int,))
    node_9 = Node(9, (int,))

    # 创建 MACD 节点并连接其子节点
    node_macd_buy = Node(macd, (pd.Series, float))
    node_macd_buy.add_child(node_close)
    node_macd_buy.add_child(node_12)
    node_macd_buy.add_child(node_26)
    node_macd_buy.add_child(node_9)

    node_macd_sell = Node(macd, (pd.Series, float))
    node_macd_sell.add_child(node_close)
    node_macd_sell.add_child(node_12)
    node_macd_sell.add_child(node_26)
    node_macd_sell.add_child(node_9)

    # 创建 Cross Over 节点并连接其子节点
    node_cross_over_buy = Node(crossover, (pd.Series, bool))
    zero = Node("0", (pd.Series, float))
    node_cross_over_buy.add_child(node_macd_buy)
    node_cross_over_buy.add_child(zero)

    node_cross_over_sell = Node(crossover, (pd.Series, bool))
    node_cross_over_sell.add_child(zero)
    node_cross_over_sell.add_child(node_macd_sell)


    # 创建根节点并连接其子节点
    root1 = Node(None, None)
    root2 = Node(None, None)
    root1.add_child(node_cross_over_buy)
    root1.add_child(node_cross_over_sell)
    root2.add_child(node_cross_over_sell)
    root2.add_child(node_cross_over_buy)
    print(calculate_mo_fitness_score(root1, root2))    

