# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, Dataset

from pydebug import debuginfo
from pydebug import debuginfo
class RandomDataset(Dataset):
    debuginfo(prj='ds', info='RandomDataset init')
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    debuginfo(prj='ds', info='BoringModel init')
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)


def test_lightning_model():
    """Test that DeepSpeed works with a simple LightningModule and LightningDataModule."""

    model = BoringModel()
    trainer = Trainer(strategy=DeepSpeedStrategy(), max_epochs=1, precision=16, accelerator="gpu", devices=1)
    trainer.fit(model)

'''

(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/lightning$ python test_simple.py
ds P: 57181 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/tests/lightning/test_simple.py f: RandomDataset L#: 14 I: RandomDataset init
ds P: 57181 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/tests/lightning/test_simple.py f: BoringModel L#: 27 I: BoringModel init

'''