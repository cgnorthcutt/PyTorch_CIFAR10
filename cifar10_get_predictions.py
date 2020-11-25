#!/usr/bin/env python
# coding: utf-8

import torch
from argparse import ArgumentParser
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cifar10_models import *
import os, sys
import numpy as np
import multiprocessing

# List of pre-trained models.
pytorch_models = [
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'densenet121',
    'densenet161',
    'densenet169',
    'mobilenet_v2',
    'googlenet',
    'inception_v3',
]

# Set up argument parser
parser = ArgumentParser(description='PyTorch CIFAR-10 Inference')
parser.add_argument('data_dir', metavar='DIR',
                    help='path to CIFAR-10 dataset folder')
parser.add_argument('-w', '--weights_dir', metavar='WEIGHTS',
                    default='/datasets/datasets/cifar10/pretrained_models',
                    help='path to pre-trained weights.'
                         'Directory must contain a folder called state_dicts')
parser.add_argument('-m', '--model', metavar='MODEL', default=None,
                    choices=pytorch_models,
                    help='model architecture: ' +
                         ' | '.join(pytorch_models) +
                         ' (example: resnet50)' +
                         ' (default: Runs across all PyTorch models)')
parser.add_argument('-g', '--gpu', metavar='GPU', default=0,
                    help='int of GPU to use. Only uses single GPU.')
parser.add_argument('-b', '--batch-size', metavar='BATCHSIZE', default=32,
                    help='Number of examples to run forward pass on at once.')
parser.add_argument('-o', '--output-dir', metavar='OUTPUTDIR',
                    default="pytorch_cifar10",
                    help='directory folder to store output results in.')
parser.add_argument('--save-all-probs', action='store_true', default=False,
                    help='Store entire softmax output of all examples (100 MB)')
parser.add_argument('--save-labels', action='store_true', default=False,
                    help='Store labels')


def main(args=parser.parse_args()):
    """Select GPU and set up data loader for ImageNet val set."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global device
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])
    test_dataset = CIFAR10(
        root=args.data_dir,
        train=False,
        transform=test_transformer,
        download=True,
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, multiprocessing.cpu_count() - 2),
        pin_memory=True,
    )
    # Run forward pass inference on all models for all examples in val set.
    models = pytorch_models if args.model is None else [args.model]
    for model in models:
        process_model(model, dataloader, args.weights_dir, args.output_dir,
                      args.save_all_probs, args.save_labels, )


def process_model(
        model_name,
        loader,
        weights_dir,  # Stores pre-trained weights
        out_dir="pytorch_cifar10",
        save_all_probs=False,
        save_labels=False,
):
    """Actual work is done here. This runs inference on a pyTorch model,
    using the pyTorch batch loader.

    Top5 predictions and probabilities for each example for the model
    are stored in the output directory."""

    # Load PyTorch model pre-trained on ImageNet
    model = eval("{}(pretrained=True, weights_dir=weights_dir)".format(model_name))
    # Send the model to GPU/CPU
    model = model.to(device)
    wfn_base = os.path.join(out_dir, model_name + "_pytorch_cifar10_")
    probs, labels = [], []

    # Inference, with no gradient changing
    model.eval()  # set model to inference mode (not train mode)
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            print("\r{} completed: {:.2%}".format(
                model_name, i / len(loader)), end="")
            sys.stdout.flush()
            out = torch.nn.functional.softmax(model(x_val.to(device)), dim=1)
            probs.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
            labels.append(y_val)

    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    if save_labels:
        np.save(wfn_base + "labels.npy", labels.astype(int))
    if save_all_probs:
        np.save(wfn_base + "probs.npy", probs.astype(np.float16))

    # Extract top 5 predictions for each example
    n = 5
    top = np.argpartition(-probs, n, axis=1)[:, :n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top]
    right1 = sum(top[range(len(top)), np.argmax(top_probs, axis=1)] == labels)
    acc1 = right1 / float(len(labels))
    count5 = sum([labels[i] in row for i, row in enumerate(top)])
    acc5 = count5 / float(len(labels))
    print('\n{}: acc1: {:.2%}, acc5: {:.2%}'.format(model_name, acc1, acc5))

    # Save top 5 predictions and associated probabilities
    np.save(wfn_base + "top5preds.npy", top)
    np.save(wfn_base + "top5probs.npy", top_probs.astype(np.float16))


if __name__ == '__main__':
    main()
