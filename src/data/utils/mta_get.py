import pickle

def mta_get(name):
    done = False
    loss_avgs = []
    loss_maxs = []
    val_avgs = []
    val_maxs = []
    count = 0
    with open('../run_' + name + '/metadata.pickle', 'rb') as f:
        while not done:
            try:
                loss_avg, loss_max, val_avg, val_max = pickle.load(f)
                loss_avgs.append(loss_avg)
                loss_maxs.append(loss_max)
                val_avgs.append(val_avg)
                val_maxs.append(val_max)
                count += 1
            except EOFError:
                done = True

    return loss_avgs, loss_maxs, val_avgs, val_maxs, count

def fromLine():
    print("Enter run name")
    return mta_get(input())

if __name__ == '__main__':
    pass
