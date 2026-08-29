import os, glob, torch
from PIL import Image
from torchvision import transforms
from evaluation.m3_score_v2 import M3V2Metric

def load(d, n, rec):
    paths=[]
    for e in ("*.png","*.jpg","*.jpeg"):
        paths += glob.glob(os.path.join(d,"**",e),recursive=True) if rec else glob.glob(os.path.join(d,e))
    paths=sorted(set(paths))[:n]
    t=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    return torch.stack([t(Image.open(p).convert("RGB")) for p in paths])

dev="cuda" if torch.cuda.is_available() else "cpu"
real=load("data_mri/brats_axial_multislice",40,True)
gen =load("output/generated_500_best",40,False)
m=M3V2Metric(device=dev, backbone_id="microsoft/rad-dino", cka_threshold=0.80, seed=42)
m.prune_layers_via_cka(real[:20])
r1=m(real,gen)
r2=m(real,gen)
print("active_layers:", m.active_layers)
print(f"run1 m3={r1['m3_score']:.10f}  weights={ {k:round(v,5) for k,v in r1['layer_weights'].items()} }")
print(f"run2 m3={r2['m3_score']:.10f}  weights={ {k:round(v,5) for k,v in r2['layer_weights'].items()} }")
print("IDENTICAL score:", r1['m3_score']==r2['m3_score'])
print("IDENTICAL weights:", r1['layer_weights']==r2['layer_weights'])
