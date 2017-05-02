#!/bin/sh

python ../../mxnet/tools/im2rec.py --list True --recursive True \
    --train-ratio 0.9 train_fine data/cifar100/cifar-100-python/fine/train/
python ../../mxnet/tools/im2rec.py --num-thread 4 --pass-through 1 \
    train_fine data/cifar100/cifar-100-python/fine/train

python ../../mxnet/tools/im2rec.py --list True --recursive True \
    --train-ratio 0.9 test_fine data/cifar100/cifar-100-python/fine/test/
python ../../mxnet/tools/im2rec.py --num-thread 4 --pass-through 1 \
    test_fine data/cifar100/cifar-100-python/fine/test

