import csv
import glob
import os
import torch
torch.set_printoptions(precision=10)


def load_data(path, data):
    files = list(filter(os.path.isfile, glob.glob(path)))
    files.sort(key=lambda x: os.path.getmtime(x))
    for file in files:
            tensor = torch.load(file)
            file_names = file.split('/')
            data_dict={}
            data_dict["pass"] = file_names[-3]
            data_dict["epoch"] = file_names[-2]
            data_dict["tensor"] = file_names[-1]
            data_dict["value"] = tensor
            data.append(data_dict)


def write_to_csv(name):
    with open(f'{name}_tensors.csv', 'w', newline='') as csvfile:
        fieldnames = ['pass', 'epoch', 'tensor', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        

for mode in ["lazy", "eager"]:
    path =f"/root/eager_vs_lazy_loss/{mode}/*/*/*"
    data=[]
    load_data(path, data)
    write_to_csv(mode)