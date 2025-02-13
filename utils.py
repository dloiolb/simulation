import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from statistics import mean
import math
from colorama import Fore

def generate_data(p, size, num_samples, csv):

  data = []

  for sample in range(num_samples):
    bernoulli_data = np.random.binomial(1,p,size)
    bernoulli_data = 2 * bernoulli_data - 1
    cumulative = np.cumsum(bernoulli_data)
    for time, (outcome, cumulative) in enumerate(zip(bernoulli_data, cumulative)):
      data.append([sample, time, outcome, cumulative])

  df_bernoulli = pd.DataFrame(
    data,columns=[
    "sample",
    "time", 
    "outcome",
    "cumulative"
    ]
  )

  df_bernoulli.to_csv(csv, index=False)

def display_data(num_samples, csv):
  df = pd.read_csv(csv)

  cols = (num_samples + 1) // 3
  rows = 3

  fig, axes = plt.subplots(rows, cols, figsize = (24, 2 * rows))

  axes = axes.flatten()
  
  #plt.plot(df['time'], df['outcome'], marker='o', linestyle='-', color='b')
  for i in range(num_samples):
    sample_data = df[df['sample'] == i]
    
    ax = axes[i]

    ax.plot(sample_data['time'], sample_data['cumulative'], label=f"Sample {i}", linestyle='-', color='b', alpha=0.7)
  
    ax.set_title(f"Sample path {i}")

  # plt.title("Brownian Motion")
  #plt.xlabel("Time (Trial)")
  #plt.ylabel("Outcome (0=Failure, 1=Success)")
  for j in range(num_samples, len(axes)):
      fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()

