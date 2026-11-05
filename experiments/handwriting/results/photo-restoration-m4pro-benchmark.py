import json,time,statistics
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(4)
class Block(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.down=nn.ModuleList([Block(3,24),Block(24,48),Block(48,96)]);self.bridge=Block(96,192);self.up=nn.ModuleList([Block(288,96),Block(144,48),Block(72,24)]);self.out=nn.Conv2d(24,1,1)
 def forward(self,x):
  skips=[]
  for b in self.down:x=b(x);skips.append(x);x=F.max_pool2d(x,2)
  x=self.bridge(x)
  for b,s in zip(self.up,reversed(skips)):x=b(torch.cat([F.interpolate(x,size=s.shape[-2:],mode='nearest'),s],1))
  return self.out(x)
assert torch.backends.mps.is_available()
model=Net().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.001)
x=torch.rand(16,3,256,256,device='mps');y=(torch.rand(16,1,256,256,device='mps')>.85).float()
times=[]
for i in range(13):
 torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);z=model(x);p=z.sigmoid();loss=F.binary_cross_entropy_with_logits(z,y)+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1);loss.backward();opt.step();torch.mps.synchronize()
 if i>=3:times.append(time.perf_counter()-start)
result=dict(device='M4 Pro MPS',parameters=sum(p.numel() for p in model.parameters()),batch_size=16,image_size=256,warmup_steps=3,measured_steps=10,seconds_per_step_median=statistics.median(times),images_per_second=16/statistics.median(times),scope='prototype U-Net; FP32 forward/backward AdamW BCE+Dice; synthetic tensors; excludes augmentation IO validation',allocated_gb=torch.mps.driver_allocated_memory()/1024**3)
print(json.dumps(result,indent=2),flush=True)
open('/tmp/hanja-restoration-benchmark.json','w').write(json.dumps(result,indent=2))
