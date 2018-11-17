# Project report

## Learning algorithm

The learning algorithm used is vanilla Deep Q Learning as described in original paper. As an input the vector of state is used instead of an image so convolutional neural nework is replaced with deep neural network. The deep neural network has following layers:

- Fully connected layer - input: 37 (state size) output: 128
- Fully connected layer - input: 128 output 64
- Fully connected layer - input: 64 output: (action size)

Parameters used in DQN algorithm:

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

## Results

![results](training_plot.png)

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

### Untrained agent

![untrained](untrained_agent.gif)

### Trained agent

![trained](trained_agent.gif)

## Ideas for future work

1. Extensive hyperparameter optimization
2. Double Deep Q Networks
3. Prioritized Experience Replay
4. Dueling Deep Q Networks
5. RAINBOW Paper
6. Learning from pixels
