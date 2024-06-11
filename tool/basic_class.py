import pandas as pd
from typing import Callable, Tuple, List, Union

class Node:
    def __init__(self, data, key):
        self.data = data  # 可能是 function name 或者是 pd.Series 或者是 int
        self.key: tuple = key
        self.father: Node = None
        self.file_name: str
        self.fitness_score :List[float]= []
        self.test_fitness_score :List[float]= []
        self.finished :bool= False
        self.parameters: List[Node] = []
        self.total_nodes :int= 0

    def get_result(self):
        if isinstance(self.data, Callable):
            parameters = [p.get_result() for p in self.parameters]
            return self.data(*parameters)
        else:
            return self.data
            
    def set_source_data(self, df:pd.DataFrame):
        '''
        為了能夠實現回測不同時段的資料
        '''
        if isinstance(self.data, str):
            if self.data == "Whole":
                self.data = df
            else:
                self.data = df[self.data]
        elif isinstance(self.data, pd.Series):
            self.data = df[self.data.name]
        elif isinstance(self.data, pd.DataFrame):
            self.data = df
        elif isinstance(self.data, Callable) or self.data == None:
            for parameter in self.parameters:
                parameter.set_source_data(df)

    def count_descend(self) -> int:
        if len(self.parameters) == 0:
            return 0
        count = 1
        for child in self.parameters:
            count += child.count_descend()
        self.total_nodes = count
        return count

    def __repr__(self):
        if isinstance(self.data, Callable):
            return f"{self.data.__name__}"
        elif isinstance(self.data, int):
            return f"{self.data}"
        elif isinstance(self.data, pd.Series):
            return f"{self.data.name}"
        else:
            return "root"
    def max_depth(self):
        if not self.parameters:
            return 1
        else:
            return 1 + max(child.max_depth() for child in self.parameters)
        
    def add_child(self, child):
        self.parameters.append(child)
        child.father = self