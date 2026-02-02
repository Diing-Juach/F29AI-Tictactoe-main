# Tic-Tac-Toe as an MDP (Dynamic Programming & Reinforcement Learning)

This repository implements **Tic-Tac-Toe as a Markov Decision Process (MDP)** and applies classic **Dynamic Programming** and **Reinforcement Learning** algorithms to solve it.

The project was built for academic learning purposes and demonstrates how planning and learning agents behave in a small, fully observable, turn-based environment.

---

## Implemented Algorithms

The following agents are implemented and evaluated:

- **Value Iteration**
- **Policy Iteration**
- **Q-Learning (ε-greedy exploration)**

Each agent learns an optimal or near-optimal policy for playing Tic-Tac-Toe against a fixed opponent.

---

## Project Structure

```
src/
 ├── environment/      # Tic-Tac-Toe environment and state transitions
 ├── mdp/              # MDP definitions and transition probabilities
 ├── agents/           # Value Iteration, Policy Iteration, Q-Learning agents
 ├── policies/         # Policy representations
 └── tests/            # Unit tests for each agent
```

---

## Requirements

- **Java 11+**
- **Maven 3.8+**

---

## Build & Test

Run all tests:

```bash
mvn test
```

Build the project:

```bash
mvn package
```

---

## Key Concepts Demonstrated

- Modeling a game as an MDP
- Bellman optimality equations
- Policy evaluation and improvement
- Model-free reinforcement learning
- Exploration vs exploitation trade-offs

---

## Notes

- Training parameters (learning rate, discount factor, ε) are configurable in the agent implementations.
- Q-Learning performance improves with increased training episodes.
- The environment is deterministic aside from agent exploration.

---

## Academic Disclaimer

This project was originally developed as part of university coursework.
It is shared here **for learning and portfolio purposes only**.

---
