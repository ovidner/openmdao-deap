from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.api import SimpleGADriver, Problem, LatinHypercubeGenerator, DOEDriver
from dataclasses import dataclass

from copy import deepcopy
import random
import numpy as np
from itertools import chain
from deap import algorithms, base, tools
from deap.benchmarks import rosenbrock


class DeapDriver(Driver):
    def _declare_options(self):
        self.options.declare("container_class")

    def _get_name(self):
        return "DeapDriver"

    def _setup_driver(self, problem):
        super()._setup_driver(problem)
        self.container = self.options["container_class"](driver=self)

    def run(self):
        final_population = self.container.run_algorithm()
        # Evaluates a point in the middle of the pareto front to have one of
        # the optimal points as the final values in the model
        # self.container.evaluate(pareto_front[len(pareto_front) // 2])
        # print(pareto_front)
        return False


class Individual(list):
    def __init__(self, *args, fitness_class, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness = fitness_class()

    def __repr__(self):
        return f"Individual({super().__repr__()})"


@dataclass(frozen=True)
class DeapContainer:
    """
    An abstract class for containing the algorithm-specific logic. This is
    instantiated in the Driver's _setup_driver() function with the driver
    itself passed in as an argument.

    This object in itself should be fully stateless.

    The motivation for having this in a dedicated object is mainly that the
    Driver class is already heavily bloated.
    """

    driver: DeapDriver

    def __post_init__(self):
        # FIXME: this API is inflexible
        self.fitness_class = type(
            "Fitness",
            (base.Fitness,),
            {"weights": (-1,) * len(self.problem.model.get_objectives())},
        )

        self.design_var_shapes = {
            name: np.shape(value)
            for (name, value) in self.driver.get_design_var_values().items()
        }
        self.objective_shapes = {
            name: np.shape(value)
            for (name, value) in self.driver.get_objective_values().items()
        }
        self.constraint_shapes = {
            name: np.shape(value)
            for (name, value) in self.driver.get_constraint_values().items()
        }

        self.individual_bounds = self._individual_bounds()

    @property
    def problem(self):
        return self.driver._problem

    def individual_factory(self, *args, **kwargs):
        individual = self.individual_class(fitness_class=self.fitness_class, *args, **kwargs)
        return individual

    def _individual_bounds(self):
        design_vars = self.problem.model.get_design_vars()
        lower, upper = chain.from_iterable(
            (design_vars[key]["lower"].flat, design_vars[key]["upper"].flat)
            for key in self.design_var_shapes.keys()
        )

        return tuple(lower), tuple(upper)

    def convert_design_vars_to_individual(self, design_vars):
        """
        Converts a dict of OpenMDAO design variables into a DEAP individual.
        """
        individual = Individual(
            chain.from_iterable(
                design_vars[key].flat for key in self.design_var_shapes.keys()
            ),
            fitness_class=self.fitness_class,
        )
        return individual

    def convert_individual_to_design_vars(self, individual):
        """
        Converts a DEAP individual into a dict of OpenMDAO design variables.
        """
        ind = deepcopy(individual)

        design_vars = {}
        for name, shape in self.design_var_shapes.items():
            ind_items = np.product(shape)
            design_vars[name] = np.reshape(ind[:ind_items], shape)
            ind = ind[ind_items:]
        return design_vars

    def get_population_generator(self, count):
        return LatinHypercubeGenerator(
            samples=count, criterion="correlation", iterations=count // 10
        )

    def init_population(self, count):
        return [
            self.convert_design_vars_to_individual(dict(case))
            for case in self.get_population_generator(count)(
                self.problem.model.get_design_vars()
            )
        ]

    def evaluate(self, individual):
        pre = id(individual.fitness)
        for (name, value) in self.convert_individual_to_design_vars(individual).items():
            self.driver.set_design_var(name, value)
        assert id(individual.fitness) == pre
        with RecordingDebugging(
            self.driver._get_name(), self.driver.iter_count, self.driver
        ):
            failure_flag, abs_error, rel_error = self.problem.model._solve_nonlinear()

        self.driver.iter_count += 1
        # print(tuple(float(x) for x in self.driver.get_objective_values().values()))
        return tuple(
            chain.from_iterable(
                x.flat for x in self.driver.get_objective_values().values()
            )
        )

    def run_algorithm(self):
        raise NotImplemented("run_algorithm() method not implemented.")
