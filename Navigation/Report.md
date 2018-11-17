# Project report

## Learning algorithm

The implemented learning algorithm is based on the Deep Q Learning approach originally described in Google´s DeepMind [Nature publication : "Human-level control through deep reinforcement learning (2015)"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). The steps of the algorithm can be found in the picture below:

![Deep Q-Learning algorithm from Udacity course](./images/DQN.png)

This algorithm screenshot is taken from the [Deep Reinforcement Learning Nanodegree course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

As an input, the vector of state obtained by the sensors of the agent is employed. Therefore, it is enough to build a Neural Network with just fully connected layers followed by Rectified Linear Units (ReLUs). If images are to be used as the state-space, then a Convolutional Neural Network (CNN) needs to be designed. The deep neural network is composed by the following layers and stages:

- Fully connected layer - input: 37 (state size) | output: 64
- ReLU layer - activation function
- Fully connected layer - input: 64 |  output 64
- ReLU layer - activation function
- Fully connected layer - input: 64 | output: (action size)

Parameters used in DQN algorithm:

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

## Results

### Training

![results](training.png)

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

### Untrained vs trained agent

![untrained](untrained_agent.gif)

![trained](trained_agent.gif)

## Ideas for future work

1. Extensive hyperparameter optimization
2. Double Deep Q Networks
3. Prioritized Experience Replay
4. Dueling Deep Q Networks
5. RAINBOW Paper
6. Learning from pixels
