#!/bin/bash

# Remove old plots
rm -r out 2>/dev/null

# Run all plots
for s in *.py; do
  echo "Running $s"
  /usr/bin/env python3 $s
done
