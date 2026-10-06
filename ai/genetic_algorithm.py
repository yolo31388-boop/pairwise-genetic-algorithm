"""
游戏内遗传算法与进化系统 - 核心模块

提供完整的遗传算法能力：
- gene_encoding:      基因编码（二进制编码 / 实数编码）
- fitness_function:   适应度函数（多目标优化 / 约束处理）
- selection:          选择（轮盘赌选择 / 锦标赛选择）
- crossover_mutation: 交叉变异（单点交叉 / 位翻转变异）
- evolve_population:  种群进化（选择 + 交叉 + 变异 + 精英保留）
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import random
import time


@dataclass
class Config:
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism: int = 1
    tournament_size: int = 3
    selection_method: str = "tournament"   # "roulette" | "tournament"
    encoding: str = "binary"               # "binary" | "real"
    bits_per_gene: int = 8
    bounds: Tuple[float, float] = (-5.12, 5.12)
    penalty_factor: float = 1000.0


class GeneticAlgorithm:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.population_size = int(self.config.get("population_size", 50))
        self.crossover_rate = float(self.config.get("crossover_rate", 0.8))
        self.mutation_rate = float(self.config.get("mutation_rate", 0.1))
        self.elitism = int(self.config.get("elitism", 1))
        self.tournament_size = int(self.config.get("tournament_size", 3))
        self.selection_method = self.config.get("selection_method", "tournament")
        self.encoding = self.config.get("encoding", "binary")
        self.bits_per_gene = int(self.config.get("bits_per_gene", 8))
        self.bounds = tuple(self.config.get("bounds", (-5.12, 5.12)))
        self.penalty_factor = float(self.config.get("penalty_factor", 1000.0))
        self._state = {}
        self._history = []
        self._rng = random.Random(self.config.get("seed"))

    # ------------------------------------------------------------------
    # 基因编码：二进制编码 / 实数编码
    # ------------------------------------------------------------------
    def gene_encoding(self, genes: Sequence[float], encoding: Optional[str] = None,
                      bits_per_gene: Optional[int] = None,
                      bounds: Optional[Tuple[float, float]] = None):
        """把实数基因编码为二进制位串或实数向量。

        - encoding="binary": 每个实数基因按 bounds 线性量化为 bits_per_gene 位 0/1，
          返回扁平的位列表。
        - encoding="real":   直接返回实数基因列表（实数编码）。
        """
        encoding = encoding or self.encoding
        if encoding == "real":
            return [float(g) for g in genes]
        if encoding != "binary":
            raise ValueError(f"未知编码方式: {encoding}")

        bits_per_gene = bits_per_gene or self.bits_per_gene
        low, high = bounds or self.bounds
        max_code = (1 << bits_per_gene) - 1
        encoded: List[int] = []
        for gene in genes:
            ratio = 0.0 if high == low else (float(gene) - low) / (high - low)
            ratio = min(1.0, max(0.0, ratio))
            code = int(round(ratio * max_code))
            for shift in range(bits_per_gene - 1, -1, -1):
                encoded.append((code >> shift) & 1)
        return encoded

    def gene_decoding(self, encoded: Sequence, encoding: Optional[str] = None,
                      bits_per_gene: Optional[int] = None,
                      bounds: Optional[Tuple[float, float]] = None) -> List[float]:
        """gene_encoding 的逆操作：把编码后的个体解码回实数基因。"""
        encoding = encoding or self.encoding
        if encoding == "real":
            return [float(g) for g in encoded]
        bits_per_gene = bits_per_gene or self.bits_per_gene
        low, high = bounds or self.bounds
        max_code = (1 << bits_per_gene) - 1
        genes: List[float] = []
        for start in range(0, len(encoded), bits_per_gene):
            code = 0
            for bit in encoded[start:start + bits_per_gene]:
                code = (code << 1) | int(bit)
            genes.append(low + (high - low) * code / max_code)
        return genes

    # ------------------------------------------------------------------
    # 适应度函数：多目标优化 / 约束处理
    # ------------------------------------------------------------------
    def fitness_function(self, individual: Sequence[float],
                         objectives: Optional[Sequence[Callable[[Sequence[float]], float]]] = None,
                         weights: Optional[Sequence[float]] = None,
                         constraints: Optional[Sequence[Callable[[Sequence[float]], float]]] = None,
                         penalty_factor: Optional[float] = None) -> float:
        """评估个体适应度（越大越好）。

        - 多目标优化：objectives 为多个目标函数，按 weights 加权求和（默认等权）。
          未提供 objectives 时默认最小化平方和（Sphere 函数取负）。
        - 约束处理：constraints 中每个函数返回 g(x)，约束为 g(x) <= 0；
          违反约束按 penalty_factor 施加罚函数，降低适应度。
        """
        if objectives:
            weights = list(weights) if weights else [1.0] * len(objectives)
            fitness = sum(w * obj(individual) for obj, w in zip(objectives, weights))
        else:
            fitness = -sum(float(g) ** 2 for g in individual)

        if constraints:
            penalty_factor = self.penalty_factor if penalty_factor is None else penalty_factor
            violation = sum(max(0.0, c(individual)) for c in constraints)
            fitness -= penalty_factor * violation
        return fitness

    # ------------------------------------------------------------------
    # 选择：轮盘赌选择 / 锦标赛选择
    # ------------------------------------------------------------------
    def selection(self, population: List[Sequence], fitnesses: Sequence[float],
                  method: Optional[str] = None, num_selected: Optional[int] = None,
                  tournament_size: Optional[int] = None) -> List[Sequence]:
        """按适应度从种群中选择个体。

        - method="roulette":   轮盘赌选择，适应度越高中选概率越大（自动平移负适应度）。
        - method="tournament": 锦标赛选择，每次随机抽 tournament_size 个个体取最优。
        """
        method = method or self.selection_method
        num_selected = num_selected or len(population)
        if not population:
            return []

        if method == "roulette":
            min_fit = min(fitnesses)
            shifted = [f - min_fit for f in fitnesses]
            total = sum(shifted)
            if total <= 0:
                probs = [1.0 / len(population)] * len(population)
            else:
                probs = [s / total for s in shifted]
            return self._rng.choices(list(population), weights=probs, k=num_selected)

        if method == "tournament":
            tournament_size = tournament_size or self.tournament_size
            selected = []
            for _ in range(num_selected):
                contenders = self._rng.sample(range(len(population)),
                                              min(tournament_size, len(population)))
                winner = max(contenders, key=lambda i: fitnesses[i])
                selected.append(population[winner])
            return selected

        raise ValueError(f"未知选择方式: {method}")

    # ------------------------------------------------------------------
    # 交叉变异：单点交叉 / 位翻转变异
    # ------------------------------------------------------------------
    def crossover_mutation(self, parent1: Sequence, parent2: Sequence,
                           crossover_rate: Optional[float] = None,
                           mutation_rate: Optional[float] = None,
                           encoding: Optional[str] = None) -> Tuple[List, List]:
        """对两个父代执行单点交叉与位翻转变异，返回两个子代。

        - 单点交叉：随机选一个切点，交换切点之后的基因段。
        - 位翻转变异：二进制编码下以 mutation_rate 概率翻转每一位（0<->1）；
          实数编码下以同概率对基因做高斯扰动。
        """
        crossover_rate = self.crossover_rate if crossover_rate is None else crossover_rate
        mutation_rate = self.mutation_rate if mutation_rate is None else mutation_rate
        encoding = encoding or self.encoding

        child1, child2 = list(parent1), list(parent2)
        if len(child1) >= 2 and self._rng.random() < crossover_rate:
            point = self._rng.randint(1, len(child1) - 1)
            child1, child2 = child1[:point] + child2[point:], child2[:point] + child1[point:]

        if encoding == "binary":
            child1 = [1 - g if self._rng.random() < mutation_rate else g for g in child1]
            child2 = [1 - g if self._rng.random() < mutation_rate else g for g in child2]
        else:
            span = self.bounds[1] - self.bounds[0]
            child1 = [g + self._rng.gauss(0, 0.1 * span) if self._rng.random() < mutation_rate else g
                      for g in child1]
            child2 = [g + self._rng.gauss(0, 0.1 * span) if self._rng.random() < mutation_rate else g
                      for g in child2]
        return child1, child2

    # ------------------------------------------------------------------
    # 种群进化：选择 + 交叉 + 变异 + 精英保留
    # ------------------------------------------------------------------
    def evolve_population(self, population: List[Sequence],
                          fitness_fn: Optional[Callable[[Sequence], float]] = None,
                          generations: int = 1,
                          selection_method: Optional[str] = None) -> List[Sequence]:
        """让种群进化 generations 代，返回最终种群。

        每代流程：评估适应度 -> 精英保留 -> 选择 -> 单点交叉 -> 位翻转变异。
        最优个体适应度记录到 self._history。
        """
        if fitness_fn is None:
            fitness_fn = lambda ind: self.fitness_function(ind)

        current = [list(ind) for ind in population]
        for _ in range(generations):
            fitnesses = [fitness_fn(ind) for ind in current]
            ranked = sorted(zip(fitnesses, current), key=lambda p: p[0], reverse=True)
            best_fitness = ranked[0][0]
            self._history.append({
                "generation": len(self._history),
                "best_fitness": best_fitness,
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "best_individual": list(ranked[0][1]),
            })

            next_gen = [list(ind) for _, ind in ranked[:self.elitism]]
            parents = self.selection(current, fitnesses, method=selection_method,
                                     num_selected=len(current))
            idx = 0
            while len(next_gen) < len(current):
                child1, child2 = self.crossover_mutation(parents[idx % len(parents)],
                                                         parents[(idx + 1) % len(parents)])
                next_gen.append(child1)
                if len(next_gen) < len(current):
                    next_gen.append(child2)
                idx += 2
            current = next_gen

        self._state["population"] = current
        return current

    # ------------------------------------------------------------------
    # 生命周期接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        population = self._state.get("population")
        if population:
            self._state["population"] = self.evolve_population(population, generations=1)

    def reset(self):
        self._state = {}
        self._history = []

    @property
    def history(self) -> List[Dict]:
        return list(self._history)
