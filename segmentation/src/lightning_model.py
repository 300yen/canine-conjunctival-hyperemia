import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex, Accuracy

# Support both direct imports and imports through the src package.
try:
    from networks import create_model
except ImportError:
    from src.networks import create_model

class ConjunctivaSegmentationModule(pl.LightningModule):
    def __init__(self, config):
        """
        config: jsonから読み込んだ辞書データ
        """
        super().__init__()
        # Store model configuration in checkpoints.
        self.save_hyperparameters()
        self.config = config

        # ---------------------------------------------------
        # 1. Initialize the model from the configuration.
        # ---------------------------------------------------
        model_name = config['model']['name']      # 例: 'albb' (ANetB相当)
        num_classes = config['dataset']['num_classes']
        
        # networks.py の関数を使ってモデルを実体化
        self.net, self.model_name_str = create_model(model_name, num_classes=num_classes)

        # ---------------------------------------------------
        # 2. 損失関数と評価指標の定義
        # ---------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        
        # 評価指標 (Accuracy, Jaccard/IoU)
        self.metric_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.metric_jac = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        
        loss = self.criterion(y, t)
        acc = self.metric_acc(y, t)
        jac = self.metric_jac(y, t)

        # ログ記録 (TensorBoard用)
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

        # Aggregate validation metrics by epoch.
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_jac', jac, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        # Configから学習率などを読み込む
        lr = self.config['optimizer']['lr']
        optimizer_type = self.config['optimizer']['type']

        # Use Adam; unrecognized optimizer names also fall back to Adam.
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            # デフォルト
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
        return optimizer