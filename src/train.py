import argparse

GAMES = {
    'frolake': 'FrozenLake-v0'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Used to train a network on the selected game \
                                     and resume run if an existing run name is given")
    parser.add_argument('game', help="Name of game to play, can come from: " + str(list(GAMES.keys())))
    parser.add_argument('run', help="Name of run to resume")
    parser.add_argument('-r', action='store_true', help="Whether to render images as game plays")
    args = parser.parse_args()

    from train_model import train_model
    train_model(GAMES[args.game.lower()], args.run, render=args.r)
