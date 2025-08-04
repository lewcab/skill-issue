import pandas as pd
from pygad import GA
from sklearn.ensemble import GradientBoostingClassifier

from constants import SEED, LOSSES, CRITERIA
from models.model import Model


class GBC(Model):
    def __init__(self, x: pd.DataFrame, y: pd.Series, search_space: list):
        """
        Initializes the GBC class.
        This class serves as a base for Gradient Boosting Classifier models.
        """
        super().__init__(GradientBoostingClassifier, x, y)
        self.search_space = search_space
        self.ga = GA(
            fitness_func=self.fitness_function,
            num_genes=len(self.search_space),
            gene_space=self.search_space,
            num_generations=10,
            num_parents_mating=5,
            sol_per_pop=10,
            mutation_probability=0.2,
            random_seed=SEED
        )

    def run(self):
        """
        Runs the genetic algorithm to optimize the model parameters.
        """
        self.ga.run()

    def fitness_function(self, ga_instance, solution, solution_idx):
        params = {
            'loss': LOSSES[int(solution[0])],
            'learning_rate': solution[1],
            'n_estimators': int(solution[2]),
            'subsample': solution[3],
            'criterion': CRITERIA[int(solution[4])],
            'max_depth': int(solution[5]),
            'min_samples_split': int(solution[6]),
            'min_samples_leaf': int(solution[7]),
            'max_features': int(solution[8]),
        }

        fitness = self.train_val_split(params)
        print(f"({solution_idx})\n\tEvaluated parameters: {params}\n\tAccuracy: {fitness:.4f}")

        params['accuracy'] = fitness
        self.solutions.append(params)

        return fitness

