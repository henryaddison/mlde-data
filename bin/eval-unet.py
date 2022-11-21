import os, sys
dir2 = os.path.abspath('unet/unet')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
import unet
import torch

model = torch.load('model.pth')

model.eval()
with torch.no_grad():
    test_batch = test_data.isel(latlong_ibox).isel({"time": slice(0*batch_size, (0+1)*batch_size)})
    predict_batch(test_batch)

model.train()