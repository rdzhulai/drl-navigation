## Project Overview

This project is developed as a part of the course "Intelligent Systems and Mobile Robotics" under the guidance of the Department of Cybernetics and Artificial Intelligence. The main objective is to implement and optimize a Deep Reinforcement Learning (DRL) algorithm for autonomous robot navigation in environments with static and dynamic obstacles. The project focuses on minimizing collisions and improving navigation efficiency using a Deep Q-Network (DQN).

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Description](#problem-description)
3. [Solution Approach](#solution-approach)
4. [Robot Model](#robot-model)
5. [Simulation and Training](#simulation-and-training)
6. [Experiments and Evaluation](#experiments-and-evaluation)
7. [Conclusion](#conclusion)
8. [Key Terms](#key-terms)
9. [Authors](#authors)
10. [References](#references)

## Introduction

Navigation of autonomous robots is a critical component of various technological applications, from supply chain logistics to exploration of unknown terrains. The goal is to enable robots to move efficiently from a starting point to a destination while avoiding static and dynamic obstacles. Recent advancements have highlighted the potential of Deep Reinforcement Learning (DRL) to enhance this capability by allowing robots to learn from experiences and improve their decision-making processes.

## Problem Description

### Problem
The environment is represented as a 2D Cartesian plane, with each object modeled geometrically as a circle. The robot and other entities in the environment are characterized by their positions, velocities, and other relevant parameters. The objective is to find an optimal control strategy that maximizes cumulative rewards over time, considering factors like action intervals and discount factors.

### Solution Method
We employ the Deep Q-Network (DQN) architecture, a powerful tool in reinforcement learning, to iteratively learn from environmental feedback and adjust its decision-making process. This involves training the DQN model using the "Experience Replay" algorithm, which allows the robot to learn from past experiences and refine its strategies over time.

## Robot Model

The robot model is developed in the Webots simulation environment. The key aspects include:
- Sensor data processing
- Neural network model for decision-making
- Reward function definition
- Training and evaluation setup

For detailed information on the robot model, please refer to section 4 of the article [Link to PDF](./article/article.pdf).

## Simulation and Training

### Scenarios
The robot is trained and evaluated across various scenarios:
1. **No obstacles**
2. **Simple static obstacles**
3. **Static obstacles of varying sizes**
4. **Complex static obstacles**
5. **Simple dynamic obstacles**
6. **Complex static and dynamic obstacles**

For a comprehensive description of each scenario, see section 5 of the article [Link to PDF](./article/article.pdf).

### Training Process
The DQN model is trained using reinforcement learning techniques, focusing on maximizing cumulative rewards by avoiding collisions and navigating efficiently. Details of the training process are elaborated in section 6 of the article [Link to PDF](./article/article.pdf).

## Experiments and Evaluation

### Experimental Setup
The robot is tested in different scenarios to assess its performance. Key metrics include navigation success rate, collision rate, and efficiency of the path taken.

### Results
Results are analyzed to determine the effectiveness of the DRL approach in improving autonomous navigation in complex environments. For experimental results and analysis, refer to section 7 of the article [Link to PDF](./article/article.pdf).

## Conclusion

This project demonstrates the potential of Deep Reinforcement Learning in enhancing autonomous robot navigation. The DQN model successfully navigates through complex environments, reducing collision rates and improving overall efficiency. For a detailed discussion, see the conclusion section of the article [Link to PDF](./article/article.pdf).

## Key Terms

- **Deep Reinforcement Learning (DRL)**: A field of machine learning combining deep learning techniques with reinforcement learning methods to create agents capable of making decisions in dynamic environments.
- **Markov Decision Process (MDP)**: A mathematical model used to describe decision-making problems where outcomes are partly random and partly under the control of a decision-maker.
- **Deep Q-Network (DQN)**: An algorithm in machine learning that combines deep neural networks with Q-learning, a reinforcement learning method, to approximate Q-values for different actions and states.
- **Experience Replay**: A technique in reinforcement learning where past experiences are stored and randomly sampled to train the model, improving learning stability and efficiency.

## Authors

- **Ivan Tkachenko**
- **Roman Dzhulai**
- **Dmytro Varich**
- **Anfisa Konycheva**

## References

For more details, please refer to the full article titled "Obstacle Avoidance using Deep Reinforcement Learning" available in the repository:
- Tkachenko, I., Dzhulai, R., Varich, D., & Konycheva, A. (2023). *Obstacle Avoidance using Deep Reinforcement Learning*. Department of Cybernetics and Artificial Intelligence, Technical University of Košice. [Link to PDF](./article/article.pdf)
