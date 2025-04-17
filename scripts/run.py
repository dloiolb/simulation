import yaml

from simulation.utils import generate_data, generate_one, display_data, display_sequence, display_one_colors

def load_config(config_file="config/generate.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
  config = load_config()
  num_samples = config['num_samples']
  p = config['p']
  size = config['size']
  k = config['k']
  T = config['T']
  csv = config['csv']
  csv2 = config['csv2']
  csv3 = config['csv3']
  
  # generate_data(p, size, num_samples,csv)
  #generate_one(p, size, csv2,'a')
  #generate_one(p, size, csv3,'b')
  
  # display_data(k, T, num_samples,csv)
  display_sequence(5, 3, T,csv)
  # display_one_colors(1000,num_samples,csv2,'a')
  # display_one_colors(1000,num_samples,csv3,'b')

if __name__ == "__main__":
  main()
