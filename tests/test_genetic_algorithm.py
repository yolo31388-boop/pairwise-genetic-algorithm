"""
游戏内遗传算法与进化系统 - 验收测试
运行方式：python -m pytest tests/test_genetic_algorithm.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm:
    def setup_method(self):
        self.system = GeneticAlgorithm({"seed": 42})

    # ---- 基因编码 ----
    def test_case_01(self):
        """二进制编码：每个基因量化为指定位数的 0/1 位串"""
        encoded = self.system.gene_encoding([0.0, 1.0], encoding="binary",
                                            bits_per_gene=8, bounds=(0.0, 1.0))
        assert len(encoded) == 16
        assert all(bit in (0, 1) for bit in encoded)
        assert encoded[:8] == [0] * 8
        assert encoded[8:] == [1] * 8

    def test_case_02(self):
        """二进制编码可解码回原基因（往返近似）"""
        genes = [-3.0, 0.5, 4.0]
        encoded = self.system.gene_encoding(genes, encoding="binary",
                                            bits_per_gene=12, bounds=(-5.0, 5.0))
        decoded = self.system.gene_decoding(encoded, encoding="binary",
                                            bits_per_gene=12, bounds=(-5.0, 5.0))
        for original, restored in zip(genes, decoded):
            assert abs(original - restored) < 0.01

    def test_case_03(self):
        """实数编码：基因保持为实数向量"""
        genes = [1.5, -2.25, 3.75]
        encoded = self.system.gene_encoding(genes, encoding="real")
        assert encoded == genes
        assert all(isinstance(g, float) for g in encoded)

    # ---- 选择 ----
    def test_case_04(self):
        """轮盘赌选择：高适应度个体被选中概率显著更高"""
        population = [[0], [1], [2], [3]]
        fitnesses = [100.0, 1.0, 1.0, 1.0]
        selected = self.system.selection(population, fitnesses,
                                         method="roulette", num_selected=200)
        wins = sum(1 for ind in selected if ind == [0])
        assert wins > 120

    def test_case_05(self):
        """锦标赛选择：锦标赛覆盖全种群时必选最优个体"""
        population = [[0], [1], [2], [3]]
        fitnesses = [1.0, 9.0, 3.0, 2.0]
        selected = self.system.selection(population, fitnesses, method="tournament",
                                         num_selected=10, tournament_size=4)
        assert all(ind == [1] for ind in selected)

    def test_case_06(self):
        """选择返回指定数量的个体"""
        population = [[i] for i in range(10)]
        fitnesses = [float(i) for i in range(10)]
        for method in ("roulette", "tournament"):
            selected = self.system.selection(population, fitnesses,
                                             method=method, num_selected=7)
            assert len(selected) == 7

    # ---- 交叉变异 ----
    def test_case_07(self):
        """单点交叉：子代由两个父代的基因段拼接而成"""
        parent1 = [0] * 10
        parent2 = [1] * 10
        child1, child2 = self.system.crossover_mutation(
            parent1, parent2, crossover_rate=1.0, mutation_rate=0.0, encoding="binary")
        assert child1 != parent1 and child2 != parent2
        for i in range(10):
            assert sorted([child1[i], child2[i]]) == [0, 1]
        # 单点交叉：每个子代至多一次 0/1 跳变
        transitions1 = sum(1 for i in range(9) if child1[i] != child1[i + 1])
        assert transitions1 <= 1

    def test_case_08(self):
        """位翻转变异：变异率 1.0 时所有位都被翻转"""
        parent1 = [0] * 8
        parent2 = [1] * 8
        child1, child2 = self.system.crossover_mutation(
            parent1, parent2, crossover_rate=0.0, mutation_rate=1.0, encoding="binary")
        assert child1 == [1] * 8
        assert child2 == [0] * 8

    # ---- 适应度函数 ----
    def test_case_09(self):
        """多目标优化：多个目标按权重加权求和"""
        objectives = [lambda x: x[0], lambda x: x[1]]
        fitness = self.system.fitness_function([2.0, 3.0], objectives=objectives,
                                               weights=[1.0, 2.0])
        assert fitness == pytest.approx(2.0 + 2.0 * 3.0)

    def test_case_10(self):
        """约束处理：违反约束的个体适应度被惩罚"""
        objectives = [lambda x: x[0]]
        constraints = [lambda x: x[0] - 1.0]  # 约束 x[0] <= 1
        feasible = self.system.fitness_function([0.5], objectives=objectives,
                                                constraints=constraints)
        infeasible = self.system.fitness_function([5.0], objectives=objectives,
                                                  constraints=constraints)
        assert feasible == pytest.approx(0.5)
        assert infeasible < feasible

    # ---- 种群进化 ----
    def test_case_11(self):
        """种群进化：经过多代进化后最优适应度显著提升"""
        ga = GeneticAlgorithm({"seed": 7, "encoding": "real", "mutation_rate": 0.2})
        population = [[ga._rng.uniform(-5, 5) for _ in range(3)] for _ in range(30)]
        fitness_fn = lambda ind: -sum(g ** 2 for g in ind)
        before = max(fitness_fn(ind) for ind in population)
        final = ga.evolve_population(population, fitness_fn=fitness_fn, generations=40)
        after = max(fitness_fn(ind) for ind in final)
        assert after > before

    def test_case_12(self):
        """种群进化：种群规模保持不变且记录进化历史"""
        ga = GeneticAlgorithm({"seed": 1, "encoding": "real"})
        population = [[float(i)] for i in range(20)]
        final = ga.evolve_population(population, generations=10)
        assert len(final) == 20
        assert len(ga.history) == 10
        assert all("best_fitness" in record for record in ga.history)

    def test_case_13(self):
        """种群进化：轮盘赌选择同样驱动进化"""
        ga = GeneticAlgorithm({"seed": 3, "encoding": "real", "mutation_rate": 0.2})
        population = [[ga._rng.uniform(-5, 5)] for _ in range(20)]
        fitness_fn = lambda ind: -ind[0] ** 2
        before = max(fitness_fn(ind) for ind in population)
        final = ga.evolve_population(population, fitness_fn=fitness_fn,
                                     generations=30, selection_method="roulette")
        assert max(fitness_fn(ind) for ind in final) > before

    def test_case_14(self):
        """完整流程：二进制编码 + 锦标赛选择 + 单点交叉 + 位翻转变异"""
        ga = GeneticAlgorithm({"seed": 11, "encoding": "binary",
                               "bits_per_gene": 8, "bounds": (0.0, 10.0)})
        targets = (7.0, 3.0)
        # 每个个体是 2 个基因（各 8 位），目标是同时逼近 targets
        population = [[ga._rng.randint(0, 1) for _ in range(16)] for _ in range(30)]

        def fitness_fn(bits):
            values = ga.gene_decoding(bits, bits_per_gene=8, bounds=(0.0, 10.0))
            return -sum(abs(v - t) for v, t in zip(values, targets))

        before = max(fitness_fn(ind) for ind in population)
        final = ga.evolve_population(population, fitness_fn=fitness_fn, generations=50)
        after = max(fitness_fn(ind) for ind in final)
        assert after > before
        assert after > -1.0  # 最优个体已逼近目标值


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
