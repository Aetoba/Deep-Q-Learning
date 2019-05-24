import pickle

def prf_get(name):
    done = False
    rewlst = []
    epslst = []
    count = 0
    with open('../run_' + name + '/performance.pickle', 'rb') as f:
        while not done:
            try:
                rew, eps = pickle.load(f)
                epslst.append(eps)
                rewlst.append(rew)
                count += 1
            except EOFError:
                done = True

    return rewlst, epslst, count

def fromLine():
    print("Enter run name")
    return prf_get(input())

if __name__ == '__main__':
    pass