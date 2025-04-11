# An Active Inference Approach to Autonomous Navigation

This project compares Active Inference (using PYMDP) and Deep Q-Learning (DQN) approaches in a simulated navigation task using the [CoppeliaSim](https://www.coppeliarobotics.com/) environment. The study focuses on understanding how these different paradigms perform in terms of learning efficiency, adaptability, and decision-making capabilities under uncertainty.

**Note:** For a visual overview and more detailed project context, please refer to the `An Active Inference Approach to Autonomus Navigation Trifold copy.pdf` file included in this repository.

## 🏆 Recognition
This project received 2nd place in Computational Biology at the Alameda County Science and Engineering Fair.

## 🧠 Project Overview

Active Inference is a theoretical framework from computational neuroscience suggesting that biological systems minimize variational free energy (or prediction error) to maintain homeostasis and guide perception, action, and learning. This project implements both Active Inference and DQN agents to solve a navigation task, allowing for a direct comparison of their performance and behavioral characteristics.

### Research Questions
1. How does Active Inference compare with Reinforcement Learning (specifically DQN) in autonomous systems navigating unknown environments, particularly regarding:
    - Avoiding myopic behavior?
    - Balancing exploration and exploitation?
    - Mitigating uncertainty?

2. How does the Free Energy Principle, central to Active Inference, contribute to efficient autonomous navigation, especially in unknown or changing environments?

### Resources for Understanding Active Inference

- [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind) - Foundational textbook by Parr, Pezzulo, and Friston.
- [Active Inference Institute](https://www.activeinference.institute/) - An open-science institute advancing Active Inference research and application.

## 📁 Project Structure

The repository is organized as follows:

```
├── Experiment/              # Main directory for core code and simulation assets
│   ├── redspots.ipynb       # Primary Jupyter notebook for running experiments and analysis
│   ├── DQN.ipynb           # Implementation of the Deep Q-Learning agent
│   ├── Gridworld.py        # Defines the grid world environment used in simulations
│   ├── GridBoard.py        # Utilities for visualization (likely with TensorBoard or similar)
│   ├── Environment.ttt     # CoppeliaSim scene file defining the simulation environment
│   ├── data_analysis/      # Scripts and notebooks for detailed data analysis (if any)
│   ├── models/             # Saved model checkpoints for ActInf and DQN agents
│   └── results/            # Raw and processed results, plots, and logs from experiments
│
├── Extra Files/            # Archived code, previous implementations, or related utilities
│   ├── RL_Agent/           # Older/alternative Reinforcement Learning agent code
│   ├── ActInfAgent/        # Older/alternative Active Inference agent code
│   ├── collect_data.py     # Data collection utilities
│   └── coppeliasim.py      # CoppeliaSim interface code
│
├── .gitignore               # Specifies intentionally untracked files that Git should ignore
├── LICENSE                  # MIT License file governing the use of this project
├── README.md                # This file, providing an overview of the project
├── requirements.txt         # Lists the Python packages required to run the code
└── An Active Inference Approach to Autonomus Navigation Trifold copy.pdf  # Visual project summary
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt`):
  - `pymdp`
  - `torch`
  - `numpy`
  - `matplotlib`
  - `jupyter`
  - `pandas`
  - `scikit-learn`
  - `tqdm`
  - `gym` (OpenAI Gym or Gymnasium)
  - `tensorboard`

### CoppeliaSim Setup

1. Download and install [CoppeliaSim](https://www.coppeliarobotics.com/)
2. Ensure the CoppeliaSim executable is accessible in your system's PATH or configure scripts accordingly
3. Launch CoppeliaSim

### Running Experiments

1. Open the `Experiment/Environment.ttt` scene file within CoppeliaSim
2. Start the simulation in CoppeliaSim
3. Navigate to the `Experiment/` directory in your terminal
4. Launch the main Jupyter notebook:
   ```bash
   jupyter notebook redspots.ipynb
   ```
5. Run the cells within the notebook to execute the experiments and view results

**Note on `.DS_Store` files:** If you encounter `.DS_Store` files (common on macOS), they can be safely ignored or removed. They are included in the `.gitignore`.

## 📊 Results and Analysis

### Key Findings
- **Active Inference:** Demonstrated superior performance in adapting to environmental changes, effectively balancing exploration and exploitation, handling uncertainty, and potentially offering more biologically plausible navigation strategies.
- **DQN:** Showed advantages in initial learning speed and computational efficiency for simpler versions of the task, along with relative implementation simplicity.

### Project Conclusions
1. Active Inference generally outperforms DQN in navigating unknown environments due to its inherent mechanisms for avoiding myopia, balancing exploration/exploitation, and mitigating uncertainty.
2. The Free Energy Principle provides a robust theoretical foundation for efficient autonomous navigation by intrinsically balancing information seeking (exploration) and goal achievement (exploitation).

### Areas for Improvement & Future Work
1. **Code Efficiency:** Optimize simulation interactions and the core Active Inference calculations.
2. **Physical Robot Implementation:** Enhance movement accuracy and sensor integration for real-world deployment.
3. **Scalability:** Test and adapt the agents for more complex, higher-dimensional environments.
4. **Integration:** Streamline the workflow between the Python code and CoppeliaSim.
5. **Generalization:** Evaluate performance across a wider variety of tasks and environmental conditions.

## 👥 Authors

- Harshil Shah
- Satyaki Maitra
- Rohit Shenoy

Mentor: Dr. Daniel Friedman (Active Inference Institute)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Active Inference Institute for mentorship and support
- The CoppeliaSim team for providing the simulation environment
- The developers of PYMDP
- The Alameda County Science and Engineering Fair