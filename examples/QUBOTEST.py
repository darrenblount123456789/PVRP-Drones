import random
import sys
import os
import numpy as np
from collections import defaultdict
import math


# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src_dir, 'src'))
from input_CMT_dataset import create_pvrp_problem
from qubo_helper_drone_schedule import Qubo
from QiskitSolversDroneSchedule import solve_qubo
import numpy as np
from qubo_helper import Qubo
from QiskitSolvers import solve_qubo
import math

# Drone Scheduler Class
class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones):
        self.routes = routes
        self.num_routes = len(routes)
        self.costs = costs
        self.num_drones = num_drones
        self.qubo = Qubo()
        self.A, self.B, self.C = self.adjust_penalties()

    def compute_route_costs(self):
        return [sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1)) for route in self.routes]
    
    def adjust_penalties(self):
        base_A, base_B, base_C = 500, 30, 350
        A = base_A + int(5 * math.log(self.num_routes + 1)) - int(10 * math.log(self.num_drones + 1))
        B = max(5, base_B - int(2 * math.log(self.num_routes + 1)) + int(3 * math.log(self.num_drones + 1)))
        C = max(300, base_C + int(10 * math.log(self.num_routes + 1)) - int(15 * math.log(self.num_drones + 1)))
        return A, B, C


    def formulate_qubo(self):
        print(f"Formulating QUBO for {self.num_drones} drones and {self.num_routes} routes")

        ###  **Step 1: Optimize Binary Encoding**
        binary_vars_per_drone = math.ceil(math.log2(self.num_drones))  
        x = {i: [f"x_{i}_{b}" for b in range(binary_vars_per_drone)] for i in range(self.num_routes)}

        print(f"🔹 Debug: x dictionary keys = {list(x.keys())[:10]} ...")

        ###  **Step 2: Assign Each Route Exactly Once**
        constraint_penalty = 500 * max(self.num_routes, self.num_drones)  
        # constraint_penalty = 10000
        for i in range(self.num_routes):
            self.qubo.add_only_one_constraint(x[i], const=constraint_penalty)

        ###  **Step 3: Load Balancing Constraint**
        avg_routes_per_drone = self.num_routes / self.num_drones  
        penalty_weight = 2000 * (self.num_routes / self.num_drones) ** 2   # This weight dynamically scales
        # penalty_weight = 100000
        drone_load = {d: [] for d in range(self.num_drones)}

        for i in range(self.num_routes):
            # Compute binary representation
            binary_rep = sum((1 if self.qubo.get_dict().get(b, 0) == 1 else 0) * 2**idx for idx, b in enumerate(x[i]))

            if binary_rep < self.num_drones:
                drone_load[binary_rep].append(x[i])  

        #  **Apply Load Balancing Penalty**
        for d, routes in drone_load.items():
            penalty_term = len(routes) - avg_routes_per_drone  
            for route_vars in routes:
                for var in route_vars:
                    self.qubo.add((var, var), penalty_weight * (penalty_term ** 2))

        ### **Step 4: Introduce Controlled Randomness (Break Symmetry)**
        # Small random perturbations prevent the solver from always picking the same solution
        # noise_factor = 5  
        # for key in self.qubo.get_dict():
        #     self.qubo.get_dict()[key] += random.uniform(-noise_factor, noise_factor)

        ###  **Step 5: Use Sparse Data Storage**
        self.qubo.terms = {k: v for k, v in self.qubo.get_dict().items() if abs(v) > 1e-9}

        print(f"🛠 **QUBO Terms Breakdown:**")
        for key, value in list(self.qubo.get_dict().items())[:1000]:
            print(f"{key}: {value}")

        print(f"Final QUBO has {len(self.qubo.get_dict())} terms.")
        return self.qubo  





    def solve(self):
        """
        Solves the QUBO and extracts a valid drone-to-route assignment.
        """
        qubo_dict = self.formulate_qubo()  # Generate QUBO formulation
        sample = solve_qubo(qubo_dict, self.num_drones)  # Solve using QUBO solver

        #  **Step 1: Extract drone-route assignments**
        formatted_solution = {}

        for key, value in sample.items():
            if value == 1:  # Consider only active assignments
                parts = key.split('_')  # Split "x_4_2" -> ["x", "4", "2"]
                if parts[0] == 'x' and len(parts) == 3:
                    route_id = int(parts[1])  # Extract route index
                    drone_id = int(parts[2])  # Extract drone index
                    formatted_solution[route_id] = drone_id  # Assign route to drone

        #  **Step 2: Ensure All Drones Get At Least One Route**
        unassigned_routes = set(range(self.num_routes)) - set(formatted_solution.keys())
        available_drones = set(range(self.num_drones))

        # Distribute unassigned routes to the least-loaded drones
        for route in unassigned_routes:
            least_loaded_drone = min(available_drones, key=lambda d: list(formatted_solution.values()).count(d))
            formatted_solution[route] = least_loaded_drone  # Assign missing routes

        print(f"Optimized Drone Assignments: {formatted_solution}")
        return formatted_solution





if __name__ == "__main__":
    # Example VRP solution (routes assigned to vehicles initially)
    routes = [
    [0, 42, 40, 19, 41, 13, 0], 
    [0, 46, 50, 21, 34, 30, 9, 0], 
    [0, 49, 10, 39, 33, 45, 15, 0], 
    [0, 8, 26, 31, 28, 22, 32, 0], 
    [0, 27, 6, 14, 25, 0], 
    [0, 47, 4, 18, 0], 
    [0, 5, 37, 44, 17, 12, 0], 
    [0, 48, 23, 7, 43, 24, 0], 
    [0, 29, 20, 35, 36, 3, 1, 0], 
    [0, 11, 2, 16, 38, 0]
]


    problem_path = r'\tests\pvrp\p-n51-k10.vrp'
    problem, g = create_pvrp_problem(problem_path)

    amount_of_drones = 10
    for num_drones in [amount_of_drones]:  # Run for a given number of drones
        print(f"\n**Running Drone Scheduling for {num_drones} Drones** \n")

        # Initialize the scheduler with new penalties
        drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)

        # Solve QUBO problem
        solution = drone_scheduler.solve()

        print(solution)

        #  **Step 1: Map routes to drones correctly**
        drone_assignments = {i: [] for i in range(num_drones)}
        for route, drone in solution.items():
            drone_assignments[drone].append(routes[route])  # Ensure correct mapping

        #  **Step 2: Print Drone Scheduling Results**
        print("\n🛠 **Drone Scheduling Results:**")
        for drone, assigned_routes in drone_assignments.items():
            total_load = sum(
                sum(problem.costs[route[i]][route[i+1]] for i in range(len(route)-1)) 
                for route in assigned_routes
            )
            print(f"🚁 **Drone {drone}:** Routes {assigned_routes}, **Total Load = {total_load}**")

        #  **Step 3: Track total assigned routes**
        total_routes = sum(len(routes) for routes in drone_assignments.values())
        print(f"\n📊 **Total Routes Assigned in Iteration 1: {total_routes}**")

        print(f"\n✅ **Completed Runs for {num_drones} Drones** ✅\n")





       # Override penalty values
            # drone_scheduler.A = A
            # drone_scheduler.B = B
            # drone_scheduler.C = C

            

# #def adjust_penalties(num_drones, num_routes):
#     """
#     Dynamically adjusts A, B, and C based on the number of drones and routes.
    
#     Args:
#     - num_drones (int): Number of drones used in the VRP solution.
#     - num_routes (int): Number of routes to be assigned.

#     Returns:
#     - (tuple): Adjusted values of A, B, and C.
#     """
    
#     # Base values from best observations
#     base_A = 666
#     base_B = 39
#     base_C = 435

#     # Adjust A: Higher with more routes, lower with more drones
#     A = base_A + int(5 * math.log(num_routes + 1)) - int(10 * math.log(num_drones + 1))

#     # Adjust B: Lower with more routes (less penalty needed)
#     B = base_B - int(2 * math.log(num_routes + 1)) + int(3 * math.log(num_drones + 1))
#     B = max(5, B)  # Ensure B stays above a reasonable minimum

#     # Adjust C: Higher if fewer drones (to prevent overload)
#     C = base_C + int(10 * math.log(num_routes + 1)) - int(15 * math.log(num_drones + 1))
#     C = max(300, C)  # Ensure C stays above a reasonable minimum

#     return A, B, C


# class DroneQUBOScheduler:
#     def __init__(self, routes, costs, num_drones, max_time=None):
#         """
#         Initializes QUBO formulation for drone job scheduling.

#         :param routes: List of vehicle routes from VRP solution.
#         :param costs: Cost matrix representing travel times/distances.
#         :param num_drones: Number of available drones.
#         """
#         self.routes = routes
#         self.num_routes = len(routes)
#         self.costs = costs
#         self.num_drones = num_drones

#         self.qubo = Qubo()  # QUBO formulation
#         self.A = 100  # Penalty for assignment constraint
#         self.B = 50  # Penalty for balancing workload

#     def compute_route_costs(self):
#         """
#         Computes the cost (travel time) of each route based on cost matrix.
#         """
#         route_costs = []
#         for route in self.routes:
#             total_cost = sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1))
#             route_costs.append(total_cost)
#         return route_costs

#     def formulate_qubo(self):
#         """
#         Constructs the QUBO problem for drone scheduling.
#         """
#         print(f"🔹 Formulating QUBO for {self.num_drones} drones with {self.num_routes} routes")
#         route_costs = self.compute_route_costs()

#         # Define binary variables: x_{i, α} (1 if route i is assigned to drone α)
#         x = {}
#         for i in range(self.num_routes):
#             for alpha in range(self.num_drones):
#                 x[(i, alpha)] = f"x_{i}_{alpha}"

#         # Constraint 1: Each route must be assigned to exactly one drone
#         for i in range(self.num_routes):
#             self.qubo.add_only_one_constraint([x[i, alpha] for alpha in range(self.num_drones)], self.A)

#         # Constraint 2: Ensure every drone gets at least one route
#         for alpha in range(self.num_drones):
#             self.qubo.add_only_one_constraint([x[i, alpha] for i in range(self.num_routes)], self.A // 2)

#         # Workload balancing constraint: Ensure workload per drone is even
#         for alpha in range(self.num_drones):
#             for i in range(self.num_routes):
#                 self.qubo.add((x[i, alpha], x[i, alpha]), route_costs[i] ** 2)
#                 if alpha > 0:
#                     self.qubo.add((x[i, alpha], x[i, 0]), -1 * route_costs[i])  # Penalize uneven workloads

#         # Objective function: Minimize the maximum workload (M_1)
#         for alpha in range(1, self.num_drones):
#             self.qubo.add((x[0, 0], x[0, alpha]), self.B)  # Encourage balanced workloads

#         print(f"Formulated QUBO with {len(self.qubo.get_dict())} terms.")
#         return self.qubo

#     def solve(self, solver_type='cpu'):
#         print(f"Solving QUBO using solver type: {solver_type}")
#         qubo_dict = self.formulate_qubo()
#         sample = solve_qubo(qubo_dict, solver_type)

#         # Convert QUBO output into a readable schedule
#         formatted_solution = {}
#         for key, value in sample.items():
#             if value == 1:  # Only include active assignments
#                 parts = key.split('_')
#                 if parts[0] == 'x':  # Extract route-drone assignment
#                     route_id, drone_id = int(parts[1]), int(parts[2])
#                     formatted_solution[route_id] = drone_id

#         print(f"Solved QUBO with formatted solution: {formatted_solution}")
#         return formatted_solution


# if __name__ == "__main__":
#     # Example VRP solution (routes assigned to vehicles initially)
#     routes = [
#         [0, 5, 14, 0], [0, 2, 6, 0], [0, 10, 3, 8, 9, 0], 
#         [0, 13, 7, 0], [0, 1, 0], [0, 11, 4, 0], [0, 15, 12, 0]
#     ]

#     problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n16-k8.vrp'
#     problem, g = create_pvrp_problem(problem_path)

#     num_drones = 3

#     # Initialize and solve QUBO for drone job scheduling
#     drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
#     solution = drone_scheduler.solve()

#     # Print the assigned routes per drone
#     drone_assignments = {i: [] for i in range(num_drones)}
#     for route, drone in solution.items():
#         drone_assignments[drone].append(routes[route])

#     print("\n🛠 Drone Scheduling Results:")
#     for drone, assigned_routes in drone_assignments.items():
#         print(f"Drone {drone}: Routes {assigned_routes}, Total Load = {sum(sum(problem.costs[route[:-1], route[1:]]) for route in assigned_routes)}")