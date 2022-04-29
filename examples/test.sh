#!/bin/bash

# -----------------------------------------------------------------
# Line-by-line breakdown of testing example.
# -----------------------------------------------------------------

# Test the example model creates by train.sh.
    # Use the provided example cut-down Grambow dataset for test data.
    # Save testing plot to this directory.
    # Output level.

KPM test ./tm1.npz \
    ./b97d3_cut_down.csv 2773 \
    --plot_dir ./tm1_plots/cut_down \
    --verbose True