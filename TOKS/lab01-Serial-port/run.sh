#!/usr/bin/env bash

tmux \
  new-session  "sudo python setup.py /dev/tnt0 write ; read" \; \
  split-window "sudo python setup.py /dev/tnt1 read ; read" \; \
  split-window "sudo python setup.py /dev/tnt1 write ; read" \; \
  split-window "sudo python setup.py /dev/tnt0 read ; read" \; \
  select-layout even-horizontal
