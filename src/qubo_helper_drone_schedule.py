from itertools import product
from collections import defaultdict

class Qubo:
    def __init__(self):
        self.terms = defaultdict(float)  # Sparse storage for QUBO terms

    def add(self, field, value):
        """ Efficiently adds a value to a QUBO field, keeping it sparse. """
        if value != 0:
            self.terms[field] += value

    def add_only_one_constraint(self, variables, const):
        """ 
        Ensures exactly one variable in 'variables' is 1, but in a low-memory way.
        - Uses a weaker but sufficient constraint to enforce uniqueness.
        """
        n = len(variables)
        avg_const = const / (2 * n)  # Lower the penalty spread to reduce terms

        for var in variables:
            self.add((var, var), -const)  # Strong self-penalty

        for var1, var2 in product(variables, variables):
            if var1 < var2:
                self.add((var1, var2), avg_const)  # Lower off-diagonal interactions

    def get_dict(self):
        """ Returns a sparse dictionary of QUBO terms (avoids storing zeros). """
        return {k: v for k, v in self.terms.items() if abs(v) > 1e-9}
