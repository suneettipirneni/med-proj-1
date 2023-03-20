import sys

from model1 import run as run_model_1
from model2 import run as run_model_2

if __name__ == '__main__':
  task = sys.argv[1]
  model = sys.argv[2]
  if task is None or model is None:
    print("Error: usage python3 main.py <task> <model>, where task can be test or train and model can be 1 or 2")
  
  print(f"{task}ing model {model}")
  if model == "1":
    run_model_1(task)
  else:
    run_model_2(task)    
  
