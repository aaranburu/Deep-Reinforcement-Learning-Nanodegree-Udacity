# Udacity DRL Nanodegree, Project 1: Navigation
First project completed for the Deep Reinforcement Learning Nanodegree in Udacity. The task consists on learning an agent to navigate and collect as many yellow bananas and as few purple bananas as possible.

The agent navigates in a large square world, where it receives a reward of +1 and a reward of -1 for each collected yellow and purple banana, respectively. Therefore, the agent aims to maximize its score by avoiding purple bananas and collecting as many yellow bananas as possible.

In this case, the state space contains 37 dimensions that describe agent´s velocity and ray based perception of objects around the agent. Regarding the action space, the agent can perform four different actions every time step: move forward (0), move backwards (1), move left (2) and move right (3).

It is necessary to remark that the task is episodic as it lasts for a certain amount of time. The environment is considered solved when the agent is able to obtain an average score of +13 or higher over 100 consecutive episodes.
