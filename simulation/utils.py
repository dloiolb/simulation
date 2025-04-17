import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from statistics import mean
import math
from colorama import Fore

def generate_data(p, size, num_samples, csv):

  data = []

  
  for sample in range(num_samples):
    bernoulli_data = 2 * np.random.binomial(1,p,size) - 1
    cumulative = np.cumsum(bernoulli_data)# /np.sqrt(size)
    cumulative[0] = 0
    for time, (outcome, cumulative) in enumerate(zip(bernoulli_data, cumulative)):
      data.append([sample, time, outcome, cumulative])

  df = pd.DataFrame(
    data,columns=[
    "sample",
    "time", 
    "outcome",
    "cumulative"
    ]
  )

  df.to_csv(csv, index=False)

def generate_one(p, size, csv,type):

  data = []
  cumulative = np.zeros(size,dtype=int)

  if type == 'a':
    for sample in range(1):
      bernoulli_data = np.random.binomial(1,p,size)
      bernoulli_data = 2 * bernoulli_data - 1
      #cumulative = np.cumsum(bernoulli_data)
      cumulative[0] = 0
      for i in range(1,size):
        cumulative[i] = max(0,cumulative[i-1] + bernoulli_data[i])
      for time, (outcome, cumulative) in enumerate(zip(bernoulli_data, cumulative)):
        data.append([sample, time, outcome, cumulative])
  elif type == 'b':
    for sample in range(1):
      bernoulli_data = np.random.binomial(1,p,size)
      bernoulli_data = 2 * bernoulli_data - 1
      cumulative = np.cumsum(bernoulli_data)
      cumulative[0] = 0
      for i in range(1,size):
        cumulative[i] = max(0,cumulative[i])
      for time, (outcome, cumulative) in enumerate(zip(bernoulli_data, cumulative)):
        data.append([sample, time, outcome, cumulative])


  df = pd.DataFrame(
    data,columns=[
    "sample",
    "time", 
    "outcome",
    "cumulative"
    ]
  )

  df.to_csv(csv, index=False) 


def display_data(k,T, num_samples, csv):
  df = pd.read_csv(csv)

  cols = (num_samples + 1) // 3
  rows = 3

  fig, axes = plt.subplots(rows, cols, figsize = (24, 2 * rows))

  axes = axes.flatten()
  
  #plt.plot(df['time'], df['outcome'], marker='o', linestyle='-', color='b')
  for i in range(num_samples):
    sample_data = df[df['sample'] == i]
    
    ax = axes[i]
    
    scaled_time = np.linspace(0,T,k)
    cumulative_trimmed = sample_data['cumulative'][:k]/np.sqrt(k)

    ax.plot(scaled_time, cumulative_trimmed, label=f"Sample {i}", linestyle='-', color='b', alpha=0.7)
  
    ax.set_title(f"Sample path {i}")

  for j in range(num_samples, len(axes)):
      fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()

def display_sequence(number, speed, T, csv):
  df = pd.read_csv(csv)

  cols = 1
  rows = number

  fig, axes = plt.subplots(rows, cols, figsize = (4, 1 * rows))

  axes = axes.flatten()
  
  sample_data = df[df['sample'] == 0]
  for i in range(number):
    
    ax = axes[i]
    power = speed*(i+1)
    k = 2**(power)
    
    scaled_time = np.linspace(0,T,k)
    cumulative_trimmed = sample_data['cumulative'][:k] /np.sqrt(k)

    ax.plot(scaled_time, cumulative_trimmed, label=f"Sample {i+1}", linestyle='-', color='b', alpha=0.7)
  
    ax.set_ylabel(f"k=2^{power}", rotation=0, labelpad=24)
    ax.yaxis.set_label_position('right')

  for j in range(number, len(axes)):
      fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()

def display_one_colors(size,num_samples, csv,type):
  df = pd.read_csv(csv)

  #cols = (num_samples + 1) // 3
  #rows = 3
  #fig, axes = plt.subplots(rows, cols, figsize = (24, 2 * rows))
  #axes = axes.flatten()
  constant_0 = 0
  constant_a = 3
  constant_b = 7
  
  plt.figure(figsize=(20, 6))
  #time = np.linspace(0, 10, 100)
  
  #plt.plot(df['time'], df['outcome'], marker='o', linestyle='-', color='b')
  path = 0
  sample_data = df[df['sample'] == path]
  #print(sample_data)
  #ax = axes[path]
  for j in range(1,size):
    current = df[(df['sample'] == path) & (df['time'] == j)]['cumulative'].iloc[0]
    prev = df[(df['sample'] == path) & (df['time'] == j-1)]['cumulative'].iloc[0]
    #print(f"{current}, {prev}")

    if prev <= current:
      #plt.plot(j, current, 'go')
      plt.plot(sample_data['time'][j-1:j+1], sample_data['cumulative'][j-1:j+1], 'g-', linewidth=1)
    else:
      #plt.plot(j, current, 'ro')
      plt.plot(sample_data['time'][j-1:j+1], sample_data['cumulative'][j-1:j+1], 'r-', linewidth=1)
    
  # plt.plot(sample_data['time'][0:size], sample_data['outcome'][0:size], linestyle='-', color='pink', alpha=0.7)
  plt.axhline(y=constant_0, color='gray', linestyle='--', label=f"{constant_0}")
  plt.axhline(y=constant_a, color='blue', linestyle='--', label=f"a = {constant_a}")
  plt.axhline(y=constant_b, color='purple', linestyle='--', label=f"b = {constant_b}")
  if type == 'a':
    plt.title(r"Sample path (fixed $\omega$), $n\mapsto Y_n(\omega)$, where $Y_0:=0, Y_n:=\text{max}(Y_{n-1}+\xi_n,0)$, for $(\xi_n)_n$ iid, uniform $\{-1,1\}$" "\n" r"$(Y_n)_n$ is a submartingale because $E(\max(Y_{n-1}+\xi_n,0)\mid\mathcal{F}_{n-1})\geq\max(E(Y_{n-1}+\xi_n\mid\mathcal{F}_{n-1}),0)=Y_{n-1}$")
  elif type == 'b':
    plt.title(r"Sample path (fixed $\omega$), $n\mapsto Y_n(\omega)$, where $Y_0:=0, Y_n:=\text{max}(\sum_{k=1}^n\xi_n=k,0)$, for $(\xi_n)_n$ iid, uniform $\{-1,1\}$" "\n" r"$(Y_n)_n$ is a submartingale because $E(\max(\sum_{k=1}^n\xi_k,0)\mid\mathcal{F}_{n-1})\geq\max(E(\xi_n+Y_{n-1}\mid\mathcal{F}_{n-1}),0)=Y_{n-1}$")
  plt.legend()

  # plt.title("Brownian Motion")
  #plt.xlabel("Time (Trial)")
  #plt.ylabel("Outcome (0=Failure, 1=Success)")
  # for j in range(num_samples, len(axes)):
  #     fig.delaxes(axes[j])

  # plt.tight_layout()
  plt.show()

