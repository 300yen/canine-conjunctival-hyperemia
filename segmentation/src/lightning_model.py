import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex, Accuracy

# さっき作った networks.py から、モデル生成関数を読み込む
# ※ 同じフォルダ(src)にあることを想定しているわ
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
        # ハイパーパラメータを自動保存（checkpointsに残るから便利よ）
        self.save_hyperparameters()
        self.config = config

        # ---------------------------------------------------
        # 1. Configに基づいてモデル（脳みそ）を作成
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

        # Validationは epochごとの集計だけでOK
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_jac', jac, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        # Configから学習率などを読み込む
        lr = self.config['optimizer']['lr']
        optimizer_type = self.config['optimizer']['type']

        # 今はAdam固定にしておくけど、将来拡張できるようにしておくわ
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            # デフォルト
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
        return optimizer