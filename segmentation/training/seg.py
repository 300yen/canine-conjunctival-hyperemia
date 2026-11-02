from torchmetrics import JaccardIndex, Accuracy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models import AlbuNet, AlbuNetB, AlbuNetN, UNet11, UNet11B, UNet11N, UNet16, UNet16B, UNet16N


class ANet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.net = AlbuNet(num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.metric_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.metric_jac = JaccardIndex(task='multiclass', num_classes=self.num_classes)


    def forward(self, x):
        h = self.net(x)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_jac', jac, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_jac', jac, on_step=False, on_epoch=True)
        return {'val_loss':loss, 'val_acc':acc}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

class ANetB(ANet):
    def __init__(self):
        super().__init__()
        self.net = AlbuNetB(num_classes=self.num_classes)

class ANetN(ANet):
    def __init__(self):
        super().__init__()
        self.net = AlbuNetN(num_classes=self.num_classes)



class U11(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet11(num_classes=self.num_classes)

class U11B(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet11B(num_classes=self.num_classes)

class U11N(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet11N(num_classes=self.num_classes)

class U16(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet16(num_classes=self.num_classes)       

class U16B(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet16B(num_classes=self.num_classes)

class U16N(ANet):
    def __init__(self):
        super().__init__()
        self.net = UNet16N(num_classes=self.num_classes)


model_dic = {'alb': ANet,
            'albb': ANetB,
            'albn': ANetN,
            'u11': U11,
            'u11b': U11B,
            'u11n': U11N,
            'u16': U16,
            'u16b': U16B,
            'u16n': U16N,
            }


def create_model(name):
    if name in model_dic:
        model_class = model_dic[name]
        net = model_class()

        return net, model_class.__name__

    else:
        print(f"Error: {name} is not a valid model.")
        return None