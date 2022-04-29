#!/bin/bash

# -----------------------------------------------------------------
# Line-by-line breakdown of testing example.
# -----------------------------------------------------------------

# Predict new Eact values using the example model trained by train.sh.
    # File containing reactants for 5 aldol reactions.
    # File containing products for 5 aldol reactions.
    # File containing enthaplies for 5 aldol reactions.
    # Save Eact predictions to this file.
    # Output level.

KPM predict tm1.npz \
    ./aldol_reacs.xyz \
    ./aldol_prods.xyz \
    ./aldol_dH.txt \
    --outfile "aldol_predictions.txt" \
    --verbose True