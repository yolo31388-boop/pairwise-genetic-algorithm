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

    # 1. 二进制编码：基因只含 0/1，长度正确
    def test_case_01(self):
        ga = GeneticAlgorithm({"encoding": "binary", "gene_length": 8,
                               "bounds": (0.0, 1.0), "seed": 1})
        genes = ga.gene_encoding([0.5])
        assert len(genes) == 8
        assert all(bit in (0, 1) for bit in genes)

    # 2. 实数编码：基因为浮点数且在取值范围内
    def test_case_02(self):
        ga = GeneticAlgorithm({"encoding": "real", "gene_length": 5,
                               "bounds": (-2.0, 3.0), "seed": 1})
        genes = ga.gene_encoding([1.5, -1.0])
        assert genes == [1.5, -1.0]
        individual = ga.random_individual()
        assert len(individual) == 5
        assert all(-2.0 <= g <= 3.0 for g in individual)

    # 3. 二进制编码可解码还原（往返一致）
    def test_case_03(self):
        ga = GeneticAlgorithm({"encoding": "binary", "gene_length": 16,
                               "bounds": (0.0, 10.0), "seed": 1})
        genes = ga.gene_encoding([5.0])
        decoded = ga.decode(genes)
        assert len(decoded) == 1
        assert abs(decoded[0] - 5.0) < 0.01

    # 4. 轮盘赌选择：返回数量正确，且偏向高适应度个体
    def test_case_04(self):
        ga = GeneticAlgorithm({"seed": 7})
        population = [[0], [1], [2], [3]]
        fitnesses = [0.0, 0.0, 0.0, 100.0]
        selected = ga.selection(population, fitnesses, count=20,
                                method="roulette")
        assert len(selected) == 20
        assert selected.count([3]) >= 15

    # 5. 锦标赛选择：每次从候选中选出最优个体
    def test_case_05(self):
        ga = GeneticAlgorithm({"seed": 3, "tournament_size": 4})
        population = [[0], [1], [2], [3]]
        fitnesses = [1.0, 2.0, 3.0, 4.0]
        selected = ga.selection(population, fitnesses, count=30,
                                method="tournament")
        assert len(selected) == 30
        # 最优个体在锦标赛中胜率最高
        assert selected.count([3]) > selected.count([0])

    # 6. 单点交叉：子代基因来自两个父代，交叉点前缀/后缀正确
    def test_case_06(self):
        ga = GeneticAlgorithm({"crossover_rate": 1.0, "mutation_rate": 0.0,
                               "seed": 5})
        parent1 = [1] * 10
        parent2 = [0] * 10
        child1, child2 = ga.crossover_mutation(parent1, parent2)
        assert len(child1) == 10 and len(child2) == 10
        # 子代是父代的拼接：前段全 1 后段全 0（或反之）
        assert child1 == [1] * child1.index(0) + [0] * (10 - child1.index(0)) \
            if 0 in child1 else child1 == [1] * 10
        assert all(g in (0, 1) for g in child1 + child2)
        assert child1 != parent1 or child2 != parent2

    # 7. 位翻转变异：变异率 1.0 时所有位被翻转
    def test_case_07(self):
        ga = GeneticAlgorithm({"mutation_rate": 1.0, "crossover_rate": 0.0,
                               "seed": 5})
        parent = [1, 0, 1, 0, 1, 0]
        child, _ = ga.crossover_mutation(parent, parent)
        assert child == [0, 1, 0, 1, 0, 1]

    # 8. 适应度函数：能区分个体好坏（全 1 个体得分最高）
    def test_case_08(self):
        ga = GeneticAlgorithm({"encoding": "binary", "gene_length": 8,
                               "bounds": (0.0, 1.0), "seed": 1})
        best = ga.fitness_function([1] * 8)
        worst = ga.fitness_function([0] * 8)
        assert best > worst
        assert best == pytest.approx(1.0)

    # 9. 多目标优化：适应度为各目标加权求和
    def test_case_09(self):
        ga = GeneticAlgorithm({"encoding": "real", "gene_length": 2,
                               "bounds": (0.0, 10.0), "seed": 1})
        ga.objectives = [(lambda v: v[0], 1.0), (lambda v: v[1], 2.0)]
        fitness = ga.fitness_function([3.0, 4.0])
        assert fitness == pytest.approx(3.0 + 2.0 * 4.0)

    # 10. 约束处理：违反约束的个体被惩罚
    def test_case_10(self):
        ga = GeneticAlgorithm({"encoding": "real", "gene_length": 1,
                               "bounds": (0.0, 10.0), "seed": 1})
        ga.constraints = [lambda v: v[0] - 5.0]  # 要求 v[0] <= 5
        feasible = ga.fitness_function([4.0])
        infeasible = ga.fitness_function([9.0])
        assert feasible > infeasible
        assert infeasible == pytest.approx(9.0 - 1000.0 * 4.0)

    # 11. 种群进化执行了选择：历史记录中最佳适应度不低于初代平均
    def test_case_11(self):
        ga = GeneticAlgorithm({"seed": 11, "population_size": 30})
        population = ga.initialize_population()
        initial = [ga.fitness_function(ind) for ind in population]
        ga.evolve_population(generations=10)
        history = ga.history
        assert len(history) == 10
        assert history[-1]["best_fitness"] >= sum(initial) / len(initial)

    # 12. 种群进化执行了交叉：子代种群与父代不同（产生了新基因组合）
    def test_case_12(self):
        ga = GeneticAlgorithm({"seed": 12, "population_size": 20,
                               "crossover_rate": 1.0, "mutation_rate": 0.0})
        population = ga.initialize_population()
        before = {tuple(ind) for ind in population}
        evolved = ga.evolve_population(generations=1)
        after = {tuple(ind) for ind in evolved}
        assert len(evolved) == len(population)
        assert after != before  # 交叉产生了新个体

    # 13. 完整进化：多代进化后种群适应度显著提升并收敛到最优
    def test_case_13(self):
        ga = GeneticAlgorithm({"seed": 13, "population_size": 40,
                               "gene_length": 16, "mutation_rate": 0.02})
        ga.initialize_population()
        ga.evolve_population(generations=100)
        _, best_fitness = ga.best_individual()
        history = ga.history
        assert history[-1]["best_fitness"] > history[0]["best_fitness"]
        assert best_fitness == pytest.approx(1.0)  # 收敛到全 1 最优解

    # 14. 框架接口：update 推进一代、reset 清空状态
    def test_case_14(self):
        ga = GeneticAlgorithm({"seed": 14, "population_size": 10})
        ga.initialize_population()
        ga.update(0.016)
        assert len(ga.history) == 1
        ga.update(0.016)
        assert len(ga.history) == 2
        ga.reset()
        assert ga.history == []
        assert ga.best_individual() == ([], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
