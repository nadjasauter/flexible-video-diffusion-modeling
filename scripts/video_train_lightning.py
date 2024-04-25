from improved_diffusion.unet_lightning import UNetVideoModelLighning
from pytorch_lightning import Trainer

model = UNetVideoModelLighning()
trainer = Trainer()
trainer.fit(model)