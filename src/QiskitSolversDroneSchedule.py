import time
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram

# Ensure all GPUs are visible
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

# def run_qaoa_on_gpu(qubo, gpu_id, shots=256):
#     backend = AerSimulator(method="statevector", device="CPU", blocking_qubits=20)
#     sampler = BackendSampler(backend=backend, options={"shots": shots})
#     qaoa_mes = NumPyMinimumEigensolver(sampler=sampler, optimizer=SPSA(), reps=1)
#     optimizer = MinimumEigenOptimizer(qaoa_mes)
#     return optimizer.solve(qubo)

# def solve_qubo(qubo, num_drones, shots=1024):
#     X = qubo.get_dict()
#     model = QuadraticProgram("qubo")
#     var_names = set()
#     quadratic = {}
#     for (x, y), value in X.items():
#         var_names.add(x)
#         var_names.add(y)
#         quadratic[(str(x), str(y))] = value
#     for var_name in var_names:
#         model.binary_var(name=str(var_name))
#     model.minimize(quadratic=quadratic)
#     num_gpus = torch.cuda.device_count()
#     shots_per_gpu = shots // num_gpus
#     with ThreadPoolExecutor(max_workers=num_gpus) as executor:
#         futures = [executor.submit(run_qaoa_on_gpu, model, i, shots_per_gpu) for i in range(num_gpus)]
#         results = [f.result() for f in futures]
#     best_result = min(results, key=lambda r: r.fval)
#     return {index: best_result.variables_dict[str(index)] for index in var_names}



import torch
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from concurrent.futures import ThreadPoolExecutor

def run_qaoa_on_quantum(qubo, shots=256):
    """
    Runs QAOA on a quantum simulator (AerSimulator with statevector or qasm backend).
    """
    backend = AerSimulator(method="matrix_product_state")  
    sampler = BackendSampler(backend=backend, options={"shots": shots})

   
    qaoa_solver = QAOA(sampler=sampler, optimizer=SPSA(), reps=3)
    optimizer = MinimumEigenOptimizer(qaoa_solver)

    return optimizer.solve(qubo)

def solve_qubo(qubo, num_drones, shots=1024):
    """
    Solves QUBO using QAOA on a quantum simulator with batch processing.
    """
    X = qubo.get_dict()
    model = QuadraticProgram("qubo")

    var_names = set()
    quadratic = {}

    for (x, y), value in X.items():
        if x != "bias" and y != "bias":  # Ensure "bias" is not treated as a variable
            var_names.add(x)
            var_names.add(y)
            quadratic[(str(x), str(y))] = value

    var_names = list(var_names)  

    for var_name in var_names:
        model.binary_var(name=str(var_name)) 

    # Set the objective function with validated variables
    try:
        model.minimize(quadratic=quadratic)
    except KeyError as e:
        print(f" ERROR: Missing variable in QuadraticProgram: {e}")
        print("Possible Fix: Ensure all variables are declared before constraints.")
        raise  

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(run_qaoa_on_quantum, model, shots)
        result = future.result()

    return {index: result.variables_dict.get(str(index), 0) for index in var_names}
