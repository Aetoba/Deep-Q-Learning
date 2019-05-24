import pickle

def prg_get(name):
    done = False
    t_stplst = []
    stplst = []
    epilst = []
    rewlst = []
    count = 0
    with open('../run_' + name + '/progress.pickle', 'rb') as f:
        while not done:
            try:
                t_stp, epi, stp, rew = pickle.load(f)
                t_stplst.append(t_stp)
                epilst.append(epi)
                stplst.append(stp)
                rewlst.append(rew)
                count += 1
            except EOFError:
                done = True

    return t_stplst, epilst, stplst, rewlst, count

def fromLine():
    print("Enter run name")
    return prg_get(input())

if __name__ == '__main__':
    _, _, stplist, rewardlist, cnt = fromLine()
    print("Rewards:")
    print(list(zip(rewardlist, stplist)))
    print("Over " + str(cnt) + " episodes")
