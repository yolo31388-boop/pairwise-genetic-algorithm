"""
游戏内遗传算法与进化系统 - 核心模块

提供完整的遗传算法能力：
- 基因编码：二进制编码 / 实数编码
- 选择：轮盘赌选择 / 锦标赛选择
- 交叉变异：单点交叉 / 位翻转变异
- 适应度函数：多目标优化 / 约束处理
- 种群进化：选择 + 交叉 + 变异的完整进化循环
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import random
import time


@dataclass
class Config:
    population_size: int = 50
    gene_length: int = 16
    encoding: str = "binary"          # "binary" | "real"
    bounds: Tuple[float, float] = (0.0, 1.0)
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    selection: str = "tournament"     # "roulette" | "tournament"
    tournament_size: int = 3
    elitism: int = 1
    seed: Optional[int] = None


class GeneticAlgorithm:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        cfg = Config()
        for key, value in self.config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        self._cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._state = {}
        self._history = []
        # 多目标优化：[(objective_fn, weight), ...]，objective_fn(values) -> float
        self.objectives: List[Tuple[Callable[[List[float]], float], float]] = []
        # 约束处理：[constraint_fn, ...]，constraint_fn(values) -> 违反量（<=0 表示可行）
        self.constraints: List[Callable[[List[float]], float]] = []
        self.constraint_penalty = 1000.0

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进一代进化（供游戏主循环调用）。"""
        population = self._state.get("population")
        if population:
            self.evolve_population(population, generations=1)

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 基因编码：二进制编码 / 实数编码
    # ------------------------------------------------------------------
    def gene_encoding(self, values: Optional[Sequence[float]] = None,
                      encoding: Optional[str] = None) -> List:
        """把实数值编码为基因；values 为 None 时随机生成一个个体。

        - binary：每个值编码为 gene_length 位的 0/1 列表
        - real：直接使用 [0,1] 归一化浮点基因
        """
        encoding = encoding or self._cfg.encoding
        lo, hi = self._cfg.bounds
        if values is None:
            values = [self._rng.uniform(lo, hi)]
        if encoding == "binary":
            genes = []
            span = hi - lo if hi > lo else 1.0
            max_int = (1 << self._cfg.gene_length) - 1
            for value in values:
                ratio = min(max((value - lo) / span, 0.0), 1.0)
                integer = int(round(ratio * max_int))
                genes.extend((integer >> shift) & 1
                             for shift in range(self._cfg.gene_length - 1, -1, -1))
            return genes
        if encoding == "real":
            return [float(v) for v in values]
        raise ValueError(f"未知编码方式: {encoding}")

    def decode(self, genes: Sequence) -> List[float]:
        """把基因解码回实数值。"""
        lo, hi = self._cfg.bounds
        if self._cfg.encoding == "real":
            return [float(g) for g in genes]
        n = self._cfg.gene_length
        max_int = (1 << n) - 1
        values = []
        for start in range(0, len(genes), n):
            chunk = genes[start:start + n]
            integer = 0
            for bit in chunk:
                integer = (integer << 1) | int(bit)
            values.append(lo + (hi - lo) * integer / max_int)
        return values

    def random_individual(self) -> List:
        """随机生成一个个体的基因。"""
        if self._cfg.encoding == "real":
            lo, hi = self._cfg.bounds
            return [self._rng.uniform(lo, hi) for _ in range(self._cfg.gene_length)]
        return [self._rng.randint(0, 1) for _ in range(self._cfg.gene_length)]

    def initialize_population(self, size: Optional[int] = None) -> List[List]:
        size = size or self._cfg.population_size
        population = [self.random_individual() for _ in range(size)]
        self._state["population"] = population
        self._state["generation"] = 0
        return population

    # ------------------------------------------------------------------
    # 适应度函数：多目标优化 / 约束处理
    # ------------------------------------------------------------------
    def fitness_function(self, genes: Sequence) -> float:
        """评估个体适应度（越大越好）。

        - 默认目标：解码值之和（二进制编码下等价于 OneMax）
        - 多目标优化：各目标按权重加权求和
        - 约束处理：对违反约束的个体施加罚函数
        """
        values = self.decode(genes)
        if self.objectives:
            fitness = sum(weight * objective(values)
                          for objective, weight in self.objectives)
        else:
            fitness = float(sum(values))
        for constraint in self.constraints:
            violation = constraint(values)
            if violation > 0:
                fitness -= self.constraint_penalty * violation
        return fitness

    # ------------------------------------------------------------------
    # 选择：轮盘赌选择 / 锦标赛选择
    # ------------------------------------------------------------------
    def selection(self, population: List[List], fitnesses: List[float],
                  count: int, method: Optional[str] = None) -> List[List]:
        method = method or self._cfg.selection
        if method == "roulette":
            return self._roulette_selection(population, fitnesses, count)
        if method == "tournament":
            return self._tournament_selection(population, fitnesses, count)
        raise ValueError(f"未知选择方式: {method}")

    def _roulette_selection(self, population, fitnesses, count):
        # 平移适应度保证权重非负
        lowest = min(fitnesses)
        weights = [f - lowest for f in fitnesses]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(population)
            total = float(len(population))
        selected = []
        for _ in range(count):
            pick = self._rng.uniform(0, total)
            cumulative = 0.0
            for individual, weight in zip(population, weights):
                cumulative += weight
                if cumulative >= pick:
                    selected.append(list(individual))
                    break
            else:
                selected.append(list(population[-1]))
        return selected

    def _tournament_selection(self, population, fitnesses, count):
        selected = []
        size = max(2, self._cfg.tournament_size)
        for _ in range(count):
            contenders = self._rng.sample(range(len(population)),
                                          min(size, len(population)))
            best = max(contenders, key=lambda i: fitnesses[i])
            selected.append(list(population[best]))
        return selected

    # ------------------------------------------------------------------
    # 交叉变异：单点交叉 / 位翻转变异
    # ------------------------------------------------------------------
    def crossover_mutation(self, parent1: List, parent2: List) -> Tuple[List, List]:
        child1, child2 = self._single_point_crossover(parent1, parent2)
        return (self._bit_flip_mutation(child1),
                self._bit_flip_mutation(child2))

    def _single_point_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        child1, child2 = list(parent1), list(parent2)
        if len(parent1) < 2 or self._rng.random() >= self._cfg.crossover_rate:
            return child1, child2
        point = self._rng.randint(1, len(parent1) - 1)
        child1 = list(parent1[:point]) + list(parent2[point:])
        child2 = list(parent2[:point]) + list(parent1[point:])
        return child1, child2

    def _bit_flip_mutation(self, genes: List) -> List:
        mutated = list(genes)
        for i in range(len(mutated)):
            if self._rng.random() < self._cfg.mutation_rate:
                if self._cfg.encoding == "real":
                    lo, hi = self._cfg.bounds
                    mutated[i] = self._rng.uniform(lo, hi)
                else:
                    mutated[i] = 1 - int(mutated[i])
        return mutated

    # ------------------------------------------------------------------
    # 种群进化：选择 + 交叉 + 变异
    # ------------------------------------------------------------------
    def evolve_population(self, population: Optional[List[List]] = None,
                          generations: int = 1) -> List[List]:
        if population is None:
            population = self._state.get("population") or self.initialize_population()
        for _ in range(generations):
            fitnesses = [self.fitness_function(ind) for ind in population]
            best_index = max(range(len(population)), key=lambda i: fitnesses[i])
            self._history.append({
                "generation": self._state.get("generation", 0),
                "best_fitness": fitnesses[best_index],
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "best": list(population[best_index]),
            })
            # 精英保留
            ranked = sorted(range(len(population)),
                            key=lambda i: fitnesses[i], reverse=True)
            next_population = [list(population[i])
                               for i in ranked[:self._cfg.elitism]]
            # 选择 + 单点交叉 + 位翻转变异
            parents = self.selection(population, fitnesses,
                                     count=len(population))
            self._rng.shuffle(parents)
            for i in range(0, len(parents) - 1, 2):
                if len(next_population) >= len(population):
                    break
                child1, child2 = self.crossover_mutation(parents[i], parents[i + 1])
                next_population.append(child1)
                if len(next_population) < len(population):
                    next_population.append(child2)
            population = next_population
            self._state["generation"] = self._state.get("generation", 0) + 1
        self._state["population"] = population
        return population

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    def best_individual(self) -> Tuple[List, float]:
        population = self._state.get("population") or []
        if not population:
            return [], 0.0
        fitnesses = [self.fitness_function(ind) for ind in population]
        best = max(range(len(population)), key=lambda i: fitnesses[i])
        return list(population[best]), fitnesses[best]
