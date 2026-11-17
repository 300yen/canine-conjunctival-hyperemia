import argparse
import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import train_val_dataloader
from seg import create_model


def training_model(model_name, fold_num, train_loader, val_loader, save_dir, img_size=512):

    date = datetime.date.today().strftime("%Y%m%d")
    pl.seed_everything(0)
    net, model_name = create_model(model_name)

    # Keep logs separate for each fold.
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f"tb_logs_{model_name}_fold{fold_num}"
    )

    checkpoint = ModelCheckpoint(
        monitor = 'val_loss',
        dirpath = save_dir,
        # Include the fold number in checkpoint filenames.
        filename = f'{model_name}_{img_size}_fold{fold_num}_{date}' + '_{epoch:02d}_{val_loss:.4f}',
        save_top_k = 1,
        mode = 'min',
        save_weights_only = True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs = 500,
        devices=1,
        accelerator="gpu",
        deterministic=False,
        callbacks=[checkpoint, early_stopping],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(net, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one dog-level fold")
    parser.add_argument("fold", type=int, choices=range(1, 6))
    parser.add_argument("--data", default="inputs")
    parser.add_argument("--checkpoints", default="03_retrained_ckpt")
    args = parser.parse_args()
    train, validation = train_val_dataloader(batch_size=4, fold=args.fold, augmentation=True,
                                            normalization=True, size=512, base_path=args.data)
    training_model("albb", args.fold, train, validation, args.checkpoints)
