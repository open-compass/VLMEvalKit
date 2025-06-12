from vlmeval import *
from vlmeval.dataset.SFEbench import SFE

dataset_ = SFE(dataset="SFE")
dataset_.load_data()
print("Dataset loaded successfully.")
msgs = dataset_.build_prompt(0)
print(msgs)
