from datasets import load_dataset
import sys

dataset = load_dataset(sys.argv[1])
print(dataset)
