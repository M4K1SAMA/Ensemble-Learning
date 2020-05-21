from train import *

def main():
    train_X, train_Y, test_X = load_data()
    print("loaded")
    train(train_X, train_Y)
    res = test(test_X)
    print("trained")
    with open('output', 'r') as f:
        for out in res:
            f.writelines()
            f.write(str(res) + '\n')

main()