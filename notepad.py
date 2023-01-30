


import torch

T_opt_Model=torch.randn(100)


batched_inputs=[{"image":T_opt_Model, "height":1200, "width":1920}]
for x in batched_inputs:
    print(x["image"])


args=[{"image":T_opt_Model, "height":1200, "width":1920},{}]
args_dict = args[-1]
args = list(args)[:-1]
n_nonkeyword = len(args)
print(n_nonkeyword)