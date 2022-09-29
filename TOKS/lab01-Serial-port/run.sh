#!/usr/bin/env bash

sudo tmux \
  new-session  "python setup.py /dev/tnt0 write ; read" \; \
  split-window "python setup.py /dev/tnt1 read ; read" \; \
  split-window "python setup.py /dev/tnt1 write ; read" \; \
  split-window "python setup.py /dev/tnt0 read ; read" \; \
  select-layout even-horizontal
