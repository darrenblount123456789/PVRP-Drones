import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_solvers import ClarkWright
import DWaveSolvers
from input import *
from input_CMT_dataset import *

problem, graph = create_vrp_problem("tests/p/pn55.vrp")
solver = ClarkWright(problem)
solution = solver.solve()
print("Solution : ", solution.solution) 
print("Total cost : ", solution.all_weights())
# print("CHECK: ", solution.all_weights())
def plot_all_solutions2(g, solutions):
    node_positions = nx.get_node_attributes(g, "pos")
    plt.figure(figsize=(16, 8))

    for i, node in enumerate(g.nodes):
        node_pos = node_positions[node]
        if i == 0:
            plt.annotate(
                "S",
                xy=node_pos,
                xytext=(-5, 5),
                textcoords="offset points",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.5),
            )
        else:
            plt.annotate(
                i,
                xy=node_pos,
                xytext=(-5, 5),
                textcoords="offset points",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, cmap.N))

    for i, solution in enumerate(solutions):
        path_x, path_y = [], []
        path_x.append(node_positions[next(iter(g.nodes))][0])
        path_y.append(node_positions[next(iter(g.nodes))][1])
        if solution:
            for node_index in solution:
                node = list(g.nodes)[node_index]
                path_x.append(node_positions[node][0])
                path_y.append(node_positions[node][1])
            plt.plot(path_x, path_y, color=colors[i % len(colors)], label=f"Route {i+1}")

    plt.legend(loc="best")
    plt.axis("off")
    plt.show()


plot_all_solutions2(graph, solution.solution)

