from datasets import load_dataset
import sys

dataset = load_dataset(sys.argv[1])
print(dataset)
labels = dataset["train"].features[sys.argv[2]].names
print(labels)

label2id, id2label = {}, {}
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(label2id)
print(id2label)
