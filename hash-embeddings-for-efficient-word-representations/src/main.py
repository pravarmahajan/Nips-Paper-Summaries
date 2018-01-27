import progressbar
import torch
import torch.nn as nn

from torch.autograd import Variable

import args
import dataloader
import dataset
from models import HashEmbedding, StandardEmbedding, ClassifierModel


def main():
    parsed_args = args.parse_arguments()
    agg_function = torch.sum
    use_cuda = parsed_args.use_gpu and torch.cuda.is_available()
    max_len = 150
    val_frac = 0.05

    if not parsed_args.with_dict:
        dl_obj = dataloader.UniversalArticleDatasetProvider(
            parsed_args.dataset, valid_fraction=val_frac)
        dl_obj.load_data()
        train_dataloader, val_dataloader, test_dataloader = dataset.create_dataset_nodict(
            dl_obj, parsed_args.batch_size, use_cuda, max_len)
        num_classes = dl_obj.num_classes
    else:
        train_dataloader, val_dataloader, test_dataloader, num_classes = dataset.create_dataset_wdict(
            parsed_args.dataset, val_frac, parsed_args.batch_size, use_cuda, max_len)

    if parsed_args.embedding_type == "hash":
        embedding_model = HashEmbedding(parsed_args.vocab_size, parsed_args.num_hash,
                                        parsed_args.bucket, parsed_args.embedding, agg_function, use_cuda)
    else:
        embedding_model = StandardEmbedding(
            parsed_args.vocab_size, parsed_args.embedding, use_cuda)

    embedding_model = embedding_model.cuda() if use_cuda else embedding_model
    embedding_model.initializeWeights()

    model = ClassifierModel(embedding_model, num_classes,
                            parsed_args.hidden, use_cuda)
    model = model.cuda() if use_cuda else model
    model.initializeWeights()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda() if use_cuda else criterion

    prev_loss = float("inf")
    for _ in range(parsed_args.num_epochs):
        bar = progressbar.ProgressBar()
        print("Epoch = {}".format(_))

        optimizer = torch.optim.Adam(list(model.parameters())+list(embedding_model.parameters()),
                                     lr=parsed_args.learning_rate)

        for (i, d) in bar(enumerate(train_dataloader)):
            t = Variable(d[1])
            t = t.cuda() if use_cuda else t
            output = model(d[0])
            loss = criterion(output, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        loss = 0
        for (i, d) in enumerate(val_dataloader):
            t = d[1].cuda() if use_cuda else d[1]
            output = model(d[0])
            pred = output.max(1)[1].data
            correct = correct + (pred == t).sum()
            total += pred.size(0)
            loss += criterion(output, Variable(t))

        loss = loss[0].data[0]
        print("Val accuracy = {:.2f}".format(correct*100/total))
        print("Val loss = {:.2f}".format(loss))
        if prev_loss < loss:
            print("Early Stopping")
            model = prev_model
            break
        else:
            prev_loss = loss
            prev_model = model

    for (i, d) in enumerate(test_dataloader):
        t = d[1].cuda() if use_cuda else d[1]
        pred = model(d[0]).max(1)[1].data
        correct = correct + (pred == t).sum()
        total += pred.size(0)
    print("Accuracy = {:.2f}".format(correct*100/total))


main()
