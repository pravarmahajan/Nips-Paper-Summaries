import argparse


def parse_arguments():
    choices = ["agnews", "dbpedia", "yelp_pol", "yelp_full", "yahoo",
               "amazon_full", "amazon_pol"]

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("-ds", "--dataset", choices=choices, default="agnews")
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
                        action="store_true")
    parser.add_argument("-emb", "--embedding_type", choices=["std", "hash"],
                        default="hash")
    parser.add_argument("-dict", "--with_dict", action="store_true")
    args = parser.parse_args()

    args.hidden = list([int(h) for h in args.hidden])
    return args
