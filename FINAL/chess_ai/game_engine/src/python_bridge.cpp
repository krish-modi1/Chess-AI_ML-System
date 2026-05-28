#include "mcts_engine.h"

// ════════════════════════════════════════════════════════════════════════════════════
// pybind11 Python Module Bindings
// 
// Exposes MCTSEngine class to Python with callback-based inference:
// 
//   from mcts_engine_cpp import MCTSEngine
//   engine = MCTSEngine(simulations=800, batch_size=8)
//   
//   def my_inference_callback(states):
//       # states: List[ChessGame] - leaf positions to evaluate
//       # Return: (policies, values) numpy arrays
//       ...
//   
//   best_move, policy = engine.search(
//       root_state, 
//       initial_policy, 
//       initial_value, 
//       temperature,
//       seed,
//       my_inference_callback  # NEW: callback for batched inference
//   )
// ════════════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(mcts_engine_cpp, m)
{
    m.doc() = "MCTS C++ Engine - Fast tree traversal with batched neural network inference via Python callbacks";

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, int>(),
             py::arg("simulations") = 800,
             py::arg("batch_size") = 8,
             "Initialize MCTS engine with simulation count and batch size")

        .def("search", &MCTSEngine::search,
             py::arg("root_state"),
             py::arg("initial_policy"),
             py::arg("initial_value"),
             py::arg("temperature") = 1.0f,
             py::arg("seed") = 0u,
             py::arg("inference_callback"),
             R"pbdoc(
                Perform MCTS search from root_state with batched neural network inference.
                
                Args:
                    root_state: ChessGame object for the root position
                    initial_policy: numpy array (8192,) - NN policy for root
                    initial_value: float - NN value for root
                    temperature: float - temperature for move selection
                    seed: uint32 - random seed for exploration diversity
                    inference_callback: callable(List[ChessGame]) -> Tuple[np.ndarray, np.ndarray]
                        Called during search to evaluate leaf positions.
                        Input: List of ChessGame objects (leaf states to evaluate)
                        Output: Tuple of (policies, values) where:
                            - policies: np.ndarray shape (batch_size, 8192)
                            - values: np.ndarray shape (batch_size,)
                
                Returns:
                    Tuple of (best_move: str, policy_vector: np.ndarray)
             )pbdoc")

        .def_readwrite("simulations", &MCTSEngine::simulations, 
                       "Number of MCTS simulations to run")
        .def_readwrite("batch_size", &MCTSEngine::batch_size, 
                       "Batch size for parallel leaf evaluation");
}
