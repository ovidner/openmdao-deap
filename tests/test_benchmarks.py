import math
from functools import partial, reduce
from operator import mul

import numpy as np
import pytest
from deap import algorithms, benchmarks, tools, base
from openmdao.api import (
    ExplicitComponent,
    Group,
    IndepVarComp,
    Problem,
    SqliteRecorder,
    CaseReader,
)

from openmdao_deap import DeapContainer, DeapDriver
from openmdao_deap.utils import (
    hyperplane_coefficients,
    is_pareto_efficient,
    cases_to_dataframe,
)

ONES = np.ones((2,))
almost_equal = partial(np.allclose, rtol=1e-2, atol=1e-2)


def easy_peasy_dtlz1(individual, obj):
    """The DTLZ1 function with g(x) = 0."""
    g = 0
    f = [0.5 * reduce(mul, individual[: obj - 1], 1) * (1 + g)]
    f.extend(
        0.5 * reduce(mul, individual[:m], 1) * (1 - individual[m]) * (1 + g)
        for m in reversed(range(obj - 1))
    )
    return f


class Nsga2Container(DeapContainer):
    def run_algorithm(self):
        toolbox = base.Toolbox()
        toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            eta=15,
            low=self.individual_bounds[0],
            up=self.individual_bounds[1],
        )
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            eta=20,
            low=self.individual_bounds[0],
            up=self.individual_bounds[1],
            indpb=0.5,
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", self.evaluate)

        pop_size = 100
        population = self.init_population(pop_size)

        return algorithms.eaMuPlusLambda(
            population=population,
            toolbox=toolbox,
            mu=pop_size,
            lambda_=pop_size // 2,
            cxpb=0.3,
            mutpb=0.3,
            ngen=30,
        )


class GenericComponent(ExplicitComponent):
    def initialize(self):
        self.options.declare("function", default=np.sum)
        self.options.declare("x_shape", default=(1,))
        self.options.declare("f_shape", default=(1,))

    def setup(self):
        self.add_input("x", np.ones(self.options["x_shape"]) * np.nan)
        self.add_output("f", np.ones(self.options["f_shape"]) * np.nan)
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["f"] = self.options["function"](inputs["x"])
        if any(np.isnan(outputs["f"])) or any(np.isnan(inputs["x"])):
            import pdb; pdb.set_trace()


@pytest.fixture
def generic_problem():
    def generate(function, x_shape, f_shape):
        prob = Problem()
        model = prob.model = Group()
        indeps = model.add_subsystem("indeps", IndepVarComp())
        indeps.add_output("x", np.ones(x_shape) * np.nan)

        model.add_subsystem(
            "function",
            GenericComponent(function=function, x_shape=x_shape, f_shape=f_shape),
        )

        model.connect("indeps.x", "function.x")

        return prob

    return generate


def test_rosenbrock(generic_problem):
    x_shape = (2,)
    f_shape = (1,)
    prob = generic_problem(
        function=benchmarks.rosenbrock, x_shape=x_shape, f_shape=f_shape
    )

    prob.model.add_design_var(
        "indeps.x", lower=np.ones(x_shape) * -2.048, upper=np.ones(x_shape) * 2.048
    )
    prob.model.add_objective("function.f")

    prob.driver = DeapDriver(container_class=Nsga2Container)
    prob.setup()
    prob.run_driver()

    assert almost_equal(prob["function.f"], np.zeros(f_shape))
    assert almost_equal(prob["indeps.x"], np.ones(x_shape))


def test_dtlz1(generic_problem, tmpdir):
    recording_path = tmpdir.join("recording.sql")

    x_shape = (3,)
    f_shape = (3,)
    prob = generic_problem(partial(easy_peasy_dtlz1, obj=3), x_shape, f_shape)

    prob.model.add_design_var(
        "indeps.x", lower=np.zeros(x_shape), upper=np.ones(x_shape)
    )
    prob.model.add_objective("function.f")

    prob.driver = DeapDriver(container_class=Nsga2Container)
    prob.driver.add_recorder(SqliteRecorder(recording_path))
    prob.setup()
    prob.run_driver()

    cases = cases_to_dataframe(CaseReader(recording_path))
    f_values = np.stack(cases["function.f"])
    pareto_mask = is_pareto_efficient(f_values)
    pareto_frontier_f = f_values[pareto_mask]

    # The DTLZ1's Pareto frontier is a hyperplane
    assert almost_equal(
        hyperplane_coefficients(pareto_frontier_f), np.array([-1, -1, 0.5])
    )
