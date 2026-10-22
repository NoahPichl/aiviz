# Agent.md

## Overview
This project focuses on evaluating the behavior of the TiRex-2 model for time series forecasting tasks. The goal is to identify biases, error patterns, and other interesting behaviors in the model's predictions using datasets from the UCR Time Series Classification Archive.

## Objectives
1. Treat the model as a black box and analyze its behavior based on inputs and outputs.
2. Investigate specific error patterns, biases, hallucinations, or shifts in output characteristics (e.g., diversity, confidence, style).
3. Use visualization techniques to expose behaviors that cannot be understood from aggregate performance measures alone.

## Methodology
1. **Dataset**:
   - Use the UCR Time Series Classification Archive as the primary dataset.
   - Optionally, extend it to a multivariate dataset if reulsts are not satisfactory

2. **Analysis**:
   - Evaluate the model's predictions for systematic behaviors.
   - Explore the model inherent biases or failure modes
   - Use 2D projections or clustering techniques to identify patterns in the model's outputs.

3. **Visualization**:
   - create visualizations that highlight model behaviors.
   - Focus on understanding patterns in depth, such as identifying sub-clusters or distinguishing features of error-prone regions.

4. **Teams**
    - There are 2 teams, team red and team blue which are working spererateld on the problem
    - Each team should have their own ideas to complete the task

## Tools and Packages
 - `numpy`
 - `pandas`
 - `torch` 
 - `plotly`
 - `matplotlib`

 Use `uv` for python packages and project management

## Model
- **TiRex-2**: [GitHub Repository](https://github.com/NX-AI/tirex-2)

## Data
- **UCR Time Series Classification Archive**: [Dataset Link](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
