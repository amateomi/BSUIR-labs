#!/usr/bin/env bash

tmux \
  new-session  "sudo python portio.py /dev/tnt0 write ; read" \; \
  split-window "sudo python portio.py /dev/tnt1 read ; read" \; \
  split-window "sudo python portio.py /dev/tnt1 write ; read" \; \
  split-window "sudo python portio.py /dev/tnt0 read ; read" \; \
  select-layout even-horizontal
