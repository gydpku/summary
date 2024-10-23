from valid_analysis import failure_analysis
import pdb
import torch
failed_cases=torch.load('failed_cases.pt')
ooa=failure_analysis(failed_cases)
