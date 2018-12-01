# Report

## Learning algorithm

The implemented learning algorithm is based on the Actor-Critic method called Deep Deterministic Policy Gradient (DDPG) approach and originally described in Google´s DeepMind [Nature publication : "Continuous Control with Deep Reinforcement Learning (20156)"](https://arxiv.org/pdf/1509.02971.pdf). As an input, the vector of state with size 33 obtained by the sensors of each agent is employed. In total, 20 agents have been simultaneously trained in parallel to improve convergence, although the implementation works fine with a single agent as well. The steps of the complete algorithm can be found in the picture below:

![Deep Deterministic Policy Gradient (DDPG) algorithm from Google DeepMind´s paper](./images/DDPG.png)

This algorithm screenshot is taken from the [Google DeepMind´s paper](https://arxiv.org/pdf/1509.02971.pdf)

The [DDPG algorithm found on Udacity´s Deep RL repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) and applied into the Pendulum Environment has been taken as a reference. Several improvements have been applied to the original algorithm. In first place, some of the hyperparameters of the Deep Neural Network of the actor and the critic have been modified. For example, batch normalization has been used between fully connected layers in order to speed up the training. It has been also noted that simpler hidden layers produce the same effect for this specific environment. The same structure has been defined for the actor and critic DNNs:

- Fully connected layer - input: 33 (state size) | output: 128
- ReLU layer - activation function
- Batch normalization
- Fully connected layer - input: 128 |  output 128
- ReLU layer - activation function
- Fully connected layer - input: 128 | output: (action size)
- Output activation layer - tanh function

At initialization, random weights of the source network have been copied to the target network. Furthermore, the learning rate for both optimizers has been set to be the same, namely 2e-4. As proposed in the benchmark implementation, gradient in the critic network has been clipped too. Apart from that, making the agent more greedy by reducing sigma value to 0.01 in the Ornstein-Uhlenbeck noise process also resulted in a faster convergence. Finally, the algorithm has been adapted to support simultaneous training of 20 agents by adding 20 experiences to the replay buffer every time step and just updating the network sampling only 10 experiences from the buffer every 20 time steps. As a result of this enhancements, training time and convergence has been significantly reduced.

Other hyperparameters tuned and used in the DDPG algorithm:

- Number of training episodes: 1000
- Maximum steps per episode: 10000
- Number of hidden layers and units per layer of neural network: [128, 128]
- Replay buffer size: 10000
- Batch size: 128
- Gamma (discount factor): 0.99
- Adam optimizer learning rate for actor and critic: 2e-4
- Tau: 1e-3
- Weight decay: 0

## Training and Results

![results](./images/training.png)

```
Episode 100	Average Score: 0.84
Episode 200	Average Score: 3.73
Episode 300	Average Score: 7.14
Episode 400	Average Score: 10.35
Episode 500	Average Score: 12.18
Episode 540	Average Score: 13.02

Environment solved in 440 episodes!	Average Score: 13.02

Episode 545	Average Score: 13.03
Episode 546	Average Score: 13.02
Episode 559	Average Score: 13.11
Episode 560	Average Score: 13.08
...
Episode 600	Average Score: 13.23
...
Episode 700	Average Score: 14.78
...
Episode 800	Average Score: 15.28
...
Episode 900	Average Score: 15.94
...
Episode 1000	Average Score: 16.25

Environment training finished after 1000 episodes!	Average Score: 16.25
```

## Future lines

1. Further fine tuning of hyperparameters for faster training
2. Implement the D4PG algorithm from Google [DeepMind´s paper](https://openreview.net/pdf?id=SyZipzbCb).
3. Try to get similar performance results with TRPO, PPO and REINFORCE.
