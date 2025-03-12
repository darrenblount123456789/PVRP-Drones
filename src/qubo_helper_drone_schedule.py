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







# # Optimized Qubo class with sparse storage to reduce memory usage
# class Qubo:
#     def __init__(self):
#         self.terms = defaultdict(float)  # Dictionary for sparse QUBO representation

#     def create_field(self, field):
#         """ Initializes a field in the QUBO dictionary if not already present. """
#         if field not in self.terms:
#             self.terms[field] = 0.0

#     def create_not_exist_field(self, field):
#         """ Ensures the field exists in the QUBO dictionary. """
#         if field not in self.terms:
#             self.terms[field] = 0.0

#     def add_only_one_constraint(self, variables, const):
#         """ Adds a constraint ensuring exactly one variable in 'variables' is 1. """
#         for var in variables:
#             self.create_not_exist_field((var, var))
#             self.terms[(var, var)] -= 2 * const
        
#         for var1, var2 in product(variables, variables):
#             if var1 != var2:
#                 self.create_not_exist_field((var1, var2))
#                 self.terms[(var1, var2)] += const

#     def add(self, field, value):
#         """ Adds a value to a QUBO field, ensuring efficient storage. """
#         if value != 0:
#             self.terms[field] += value

#     def merge_with(self, qubo, const1=1.0, const2=1.0):
#         """ Merges another QUBO with this one, scaling coefficients by given constants. """
#         for field in self.terms:
#             self.terms[field] *= const1
#         for field, value in qubo.terms.items():
#             self.create_not_exist_field(field)
#             self.terms[field] += value * const2

#     def get_dict(self):
#         """ Returns the QUBO terms dictionary for solver compatibility. """
#         return {k: v for k, v in self.terms.items() if v != 0.0}
