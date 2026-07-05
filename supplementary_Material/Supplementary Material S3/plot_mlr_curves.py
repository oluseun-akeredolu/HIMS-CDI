#!/usr/bin/env python3
"""
HIMS_CDI_MLR Training Curves Generator
======================================

This script reads S3_loss_data.csv and generates a publication-ready plot
showing the training and validation loss dynamics for the HIMS_CDI_MLR model.

Features:
  - Faint individual seed curves (transparent lines, n=15 seeds)
  - Bold mean trend lines (training and validation)
  - Proper axis labels, legend, and title
  - Saves as both PDF and PNG

Usage:
  python plot_mlr_curves.py

Requirements:
  - pandas
  - matplotlib
  - numpy

Author: HIMS_CDI Research Team
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_mlr_training_curves(csv_path="S3_loss_data.csv", 
                              output_pdf="hims_cdi_mlr_training_curves.pdf",
                              output_png="hims_cdi_mlr_training_curves.png"):
    """
    Generate HIMS_CDI_MLR training curves from CSV data.
    
    Parameters:
    -----------
    csv_path : str
        Path to S3_loss_data.csv file
    output_pdf : str
        Output path for PDF file
    output_png : str
        Output path for PNG file
    """
    
    print("=" * 70)
    print("GENERATING HIMS_CDI_MLR TRAINING CURVES")
    print("=" * 70)
    
    # 1. Load the data
    print(f"\n✓ Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for the HIMS_CDI_MLR model
    mlr_data = df[df["model"] == "hims_cdi_mlr"].copy()
    
    print(f"  Total records for HIMS_CDI_MLR: {len(mlr_data)}")
    print(f"  Seeds: {len(mlr_data['seed'].unique())}")
    print(f"  Epochs: {sorted(mlr_data['epoch'].unique())}")
    
    # Ensure epoch is integer
    mlr_data["epoch"] = mlr_data["epoch"].astype(int)
    
    # Pivot to get train and val loss per seed and epoch
    train_pivot = mlr_data[mlr_data["loss_type"] == "train"].pivot(
        index="epoch", columns="seed", values="loss_value"
    )
    val_pivot = mlr_data[mlr_data["loss_type"] == "val"].pivot(
        index="epoch", columns="seed", values="loss_value"
    )
    
    # 2. Compute mean across seeds
    print("\n✓ Computing statistics...")
    mean_train = train_pivot.mean(axis=1)
    mean_val = val_pivot.mean(axis=1)
    
    epochs = train_pivot.index.values
    print(f"  Mean train loss range: {mean_train.min():.4f} to {mean_train.max():.4f}")
    print(f"  Mean val loss range: {mean_val.min():.4f} to {mean_val.max():.4f}")
    
    # 3. Create plot
    print("\n✓ Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Faint individual seed lines
    print(f"  - Adding {len(train_pivot.columns)} faint individual seed curves...")
    for seed in train_pivot.columns:
        ax.plot(epochs, train_pivot[seed], 
               color="blue", alpha=0.15, linewidth=0.7, label="_nolegend_")
    
    for seed in val_pivot.columns:
        ax.plot(epochs, val_pivot[seed], 
               color="orange", alpha=0.15, linewidth=0.7, label="_nolegend_")
    
    # Bold mean trend lines
    print(f"  - Adding bold mean trend lines...")
    ax.plot(epochs, mean_train, 
           color="blue", linewidth=3, label="Mean Train Loss", 
           marker='o', markersize=5)
    ax.plot(epochs, mean_val, 
           color="orange", linewidth=3, label="Mean Validation Loss", 
           marker='s', markersize=5)
    
    # Formatting
    ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=13, fontweight='bold')
    ax.set_title("HIMS_CDI_MLR Training Dynamics", fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, len(epochs) + 0.5)
    
    plt.tight_layout()
    
    # Save figure
    print(f"\n✓ Saving outputs...")
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved PDF: {output_pdf}")
    
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved PNG: {output_png}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print(f"\nThe plot is ready for inclusion in your publication.")
    print(f"Both PDF and PNG versions are available for different use cases.")


if __name__ == "__main__":
    # Run with default parameters
    plot_mlr_training_curves()
