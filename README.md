# AI in the Sciences and Engineering Final Project (ETH Zurich, Fall 2025)

This repository contains my final project for the course AI in the Sciences and Engineering (Fall 2025 at ETH Zurich).

## Overview

The project is organized into three parts:

### Part 1: Visualizing loss landscapes, PINNs vs Data-Driven
Goal: compare optimization behavior and spectral bias when solving a multiscale 2D Poisson equation with increasing frequency content.

What I did:
- Generated datasets of source terms and analytic solutions on a 2D grid for multiple complexity levels (K controls frequency content).
- Trained an MLP as a neural PDE solver in two regimes:
  - Data-Driven supervised loss against the analytical solution.
  - PINN residual loss using automatic differentiation for the PDE residual plus boundary condition penalty.
- Evaluated performance across low, medium, and high complexity settings (K = 1, 4, 16).
- Bonus: visualized local loss landscapes around converged parameters using a 2D plane projection technique (Li et al., 2018) and compared PINN vs Data-Driven geometry as K increases.

### Part 2: Training a Fourier Neural Operator (FNO) for an unknown dynamical system
Goal: learn a neural operator that maps an initial condition u0(x) to future states u(x, t) from trajectory data.

What I did:
- One-to-one training: trained an FNO to predict u(t = 1.0) from u0.
- Resolution testing: evaluated the trained model on multiple spatial resolutions (s in {32, 64, 96, 128}).
- All-to-all training: trained a time-dependent FNO using all time snapshots, with time included as conditioning, then evaluated errors at multiple time steps.
- Finetuning: tested zero-shot generalization to a shifted initial condition distribution, then finetuned on a small number of trajectories to measure transfer learning benefits.
- Bonus: trained a model from scratch on the shifted distribution and compared against finetuning.

### Part 3: Geometry-Aware Operator Transformer (GAOT), random sampling tokenization
Goal: extend GAOT beyond structured grids to better handle irregular geometries, by implementing random sampling tokenization and adaptive neighborhood aggregation.

What I did:
- Baseline: trained the official GAOT model with structured stencil grid tokenization on the Elasticity dataset and recorded test error, token count, and training time.
- Random sampling tokenization: replaced fixed mesh tokens with randomly sampled points in the domain.
- Dynamic radius strategy: implemented a local, data-driven radius per token based on neighborhood distances to maintain connectivity and coverage under random sampling.
- Bonus: explored positional encoding designs for continuous coordinates, including absolute coordinate embeddings and continuous relative bias alternatives.

## Report

- Full project report (PDF): [project_report.pdf](project_report.pdf)
