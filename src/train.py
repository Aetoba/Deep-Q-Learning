import argparse

# Whether or not to use retro package instead of gym for the environment
# Note that frame skipping is non-det in retro
use_retro = False

if use_retro:
    GAMES = {
        'spinv': 'SpaceInvaders-Atari2600',
        'brkt' : 'Breakout-Atari2600'
    }
else:
    GAMES = {
        'spinv': 'SpaceInvadersDeterministic-v4',
        'brkt' : 'BreakoutDeterministic-v4'
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Used to train a network on the selected game \
                                     and resume run if an existing run name is given")
    parser.add_argument('game', help="Name of game to play, can come from: " + str(list(GAMES.keys())))
    parser.add_argument('run', help="Name of run to resume")
    parser.add_argument('-er_size', action='store', help="Size of experience replay buffer", type=int)
    parser.add_argument('-c', action='store_true', help="Whether to capture the frames")
    parser.add_argument('-r', action='store_true', help="Whether to render images as game plays")
    parser.add_argument('-cpu', action='store_true', help="Whether to run on CPU")

    args = parser.parse_args()

    from train_model import train_model
    if args.er_size is not None:
        train_model(GAMES[args.game.lower()], args.run, use_retro,
                    capture=args.c, render=args.r, exp_replay_size=args.er_size, cpu=args.cpu)
    else:
        train_model(GAMES[args.game.lower()], args.run, use_retro,
                    capture=args.c, render=args.r, cpu=args.cpu)
