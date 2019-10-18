#!/bin/bash

python plot.py --env=CartPole-v0 --var=duration --win=10 --n_steps 1 4 9 --standalone

python plot.py --env=FrozenLake-v0 --var=G --win=100 --standalone

python plot.py --env=Copy-v0 --var=G --win=100
python plot.py --env=RepeatCopy-v0 --var=G --win=100
python plot.py --env=DuplicatedInput-v0 --var=G --win=100
python plot.py --env=Reverse-v0 --var=G --win=100
