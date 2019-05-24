# Exploring the Implementation of Deep Q-Learning to Play Atari Games

Hadrien Pouget

This is the code I wrote for my 4th year Project, on implementing Deep Q-Learning as in the 2013 DeepMind NIPS Paper.

## Breakout

It learned to play Breakout:

![BrktPrf](/src/data/not_runs/brktperformance.png)

###### Each orange point is the average performance over the last 100 training episodes with 0.05 exploration rate. Green is max performance. Blue is average human performance. The algorithm was trained for 5000 episodes.

Playing Breakout randomly yields a very low average score (roughly 1.2):

![BrktRandGif](/src/data/not_runs/brkt_rand.gif)

Our agent does much better than this:

![BrktGif](/src/data/not_runs/brkt.gif)
![BrktPreProcGif](src/data/not_runs/brkt_pre_proc.gif)

###### Left: gif of trained agent playing Breakout. Right: What agent 'sees' after preprocessing of image

## Space Invaders

It failed to achieve similar results in Space Invaders, struggling to perform better than random:
![SpInvPrf](/src/data/not_runs/spinvperformance.png)
![SpInvPrf2](/src/data/not_runs/spinv2performance.png)

###### Each point is average performance over the last 100 episodes with 0.05 exploration rate

The report contains further analysis of the results and possible issues.

## Running the code

To run the code, use `python train.py` from in './src/' and type `python train.py -h` for help. The scripts in './data/utils/' can be used to plot performance after training.

To run an agent that has already been trained on Breakout, run `python run_trained_brkt.py`.

The code was written with functionality in mind, and so could use some cleaning up!