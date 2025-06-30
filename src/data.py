import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

''' 대회용 문서 이미지 데이터를 불러와서 학습용(train)과 검증용(val)로 나누는 CustomDataModule 클래스 정의 '''

# TODO: 다음 스텝에서 실제 이미지와 라벨을 불러올 Dataset 클래스를 정의할 예정
# from torch.utils.data import Dataset
# class CustomDataset(Dataset):
#     def __init__(self, ...):
#         pass
#     def __len__(self):
#         pass
#     def __getitem__(self, idx):
#         pass

''' 모든 데이터 관리 로직(경로 설정, 데이터 분할, 로더 생성) '''
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, num_workers):
        super().__init__() # config.yaml에서 받아옴
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

    # 이 함수는 최초 1회만 호출됨. 데이터 불러오고 학습용-검증용으로 나눔
    def setup(self, stage=None):
        # 1. 전체 데이터를 불러오고 StratifiedKFold로 분할
        df = pd.read_csv(f"{self.path}/train.csv")
        # 타겟을 비율 유지하면서 5조각으로 나누고 첫번째 조각을 검증용으로 사용
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
        
        # train/validation 인덱스 분리
        # 여기서는 첫 번째 fold를 validation으로 사용
        train_idx, val_idx = next(iter(skf.split(df, df['target'])))
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 2. 데이터셋 정의 (지금은 데이터프레임만 넘겨주지만, 나중에 CustomDataset으로 교체)
        # self.train_dataset = CustomDataset(df=train_df, ...)
        # self.val_dataset = CustomDataset(df=val_df, ...)
        self.train_dataset = train_df # 임시
        self.val_dataset = val_df   # 임시
        print(f"Train/Val 데이터셋 분리 완료. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")


    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return "train_loader_placeholder" # 임시

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return "val_loader_placeholder" # 임시