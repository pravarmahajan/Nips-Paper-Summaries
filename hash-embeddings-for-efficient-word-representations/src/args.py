import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-K", "--vocab_size", help="size of the vocabulary",
                        type=int, default=10**7)
    parser.add_argument("-k", "--num_hash", help="number of hash functions",
                        type=int, default=2)
    parser.add_argument("-B", "--bucket", help="number of embedding buckets",
                        type=int, default=10**6)
    parser.add_argument("-d", "--embedding", help="embedding vector size",
                        type=int, default=20)
    parser.add_argument("-h", "--hidden", help="number of hidden units",
                        nargs="*", default=[0])
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=0.001)
    parser.add_argument("-e", "--num_epochs", help="number of epochs",
                        type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="size of a minibatch",
                        type=int, default=1024)
    parser.add_argument("-g", "--use_gpu", help="use gpu?",
                        type=bool, default=True)
    args = parser.parse_args()
    return args
