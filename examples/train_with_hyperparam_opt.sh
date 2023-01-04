#!/bin/bash

# -----------------------------------------------------------------
# Line-by-line breakdown of training example.
# -----------------------------------------------------------------

# Train a model on the reactions in the Grambow dataset.
    # Set path for model output.
    # Train on both forward and backward reactions.
    # Normalise Eact training data with Z-score normalisation.
    # Split data into train/test data by K-fold cross validation.
    # 5-fold splitting.
    # Save splitting indices to this file.
    # Seed for randomised splitting.
    # Save train/test set predictions to this directory.
    # Save test/train sets to this file.
    # Save correlation plots to this directory.
    # Neural network activation function.
    # Number of neural networks to train and average over.
    # Type of descriptor to use in creation of difference fingerprints.
    # Morgan fingerprint length.
    # Morgan fingerprint radius.
    # Output level.

KPM train ./b97d3_ORIGINAL.csv 16365 \
    --model_out ./tm1.npz \
    --train_direction both \
    --norm_type Zscore \
    --split_method cv \
    --split_num 5 \
    --split_index_path ./tm1_cv_idx.npz \
    --random_seed 1000 \
    --training_prediction_dir ./ \
    --save_test_train_path ./tm1_tt_te_data.npz \
    --plot_dir ./tm1_plots \
    --opt_hyperparams True \
    --opt_hyperparams_jobs 6 \
    --opt_hyperparams_file ./hyperparams.json
    --nn_activation_function relu \
    --nn_ensemble_size 3 \
    --descriptor_type MorganF \
    --morgan_num_bits 1024 \
    --morgan_radius 5 \
    --verbose True                               