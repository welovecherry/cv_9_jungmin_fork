import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, num_classes=17):
        super().__init__()
        # 파라미터를 저장하면 나중에 체크포인트에서 자동으로 불러올 수 있어 편리해
        self.save_hyperparameters()

        # 1. 모델 로드 (베이스라인 아이디어 적용)
        # 베이스라인에서 timm.create_model을 사용한 것과 동일한 방식 
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=self.hparams.num_classes
        )

        # 2. 손실 함수 정의 (베이스라인 아이디어 적용)
        # 베이스라인과 동일하게 CrossEntropyLoss 사용 
        self.loss_fn = nn.CrossEntropyLoss()

        # 3. 평가지표 정의 (베이스라인 아이디어 적용 + 업그레이드)
        # 대회의 핵심 지표인 Macro F1 Score 
        self.f1_score = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro')
        self.accuracy = MulticlassAccuracy(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 훈련 데이터에 대한 로직
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)

        # 훈련 과정의 손실과 정확도를 기록
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', self.f1_score(preds, targets), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(preds, targets), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # 검증 데이터에 대한 로직
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)

        # 검증 과정의 손실과 정확도를 기록
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1_score(preds, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(preds, targets), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        # 4. 옵티마이저 설정 (베이스라인 아이디어 적용)
        # 베이스라인에서 사용한 Adam 옵티마이저 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer