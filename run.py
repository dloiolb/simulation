import yaml

from utils import generate_data, display_data

def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
  config = load_config()
  num_samples = config['num_samples']
  p = config['p']
  size = config['size']
  csv = config['csv']
  
  generate_data(p, size, num_samples,csv)
  display_data(num_samples,csv)

if __name__ == "__main__":
  main()
