import pytorch_lightning as pl
import pandas as pd
import cv2  # opencv-python 패키지
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2


''' 대회용 문서 이미지를 실제로 불러오는 CustomDataset 클래스 '''
class CustomDataset(Dataset):
    """
    데이터프레임과 이미지 루트 경로를 받아, 인덱스에 해당하는 이미지와 라벨을 반환하는 클래스.
    """
    def __init__(self, df, data_root, transform=None):
        super().__init__()
        self.df = df
        self.data_root = data_root  # 'data/raw/train' 같은 이미지 폴더 경로
        self.transform = transform  # Albumentations 이미지 변환기
        
        # 데이터프레임에서 이미지 파일 이름과 라벨을 미리 리스트로 저장해두면 속도가 빨라짐.
        self.image_paths = self.df['ID'].values
        self.labels = self.df['target'].values

    def __len__(self):
        # 이 데이터셋의 총 아이템 개수를 반환.
        return len(self.df)

    def __getitem__(self, idx):
        # DataLoader가 이 함수를 호출해서 idx에 해당하는 데이터를 요청.
        
        # 1. idx에 해당하는 이미지 경로와 라벨을 가져오기
        image_path = f"{self.data_root}/{self.image_paths[idx]}"
        label = self.labels[idx]

        # 2. 이미지 파일을 실제로 읽어오기 (OpenCV 사용)
        # 이미지를 BGR 순서로 읽어오므로, 일반적인 RGB 순서로 바꿔줘야함.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 이미지 증강 및 전처리 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # 4. 이미지와 라벨을 쌍으로 묶어서 반환!
        # 이렇게 두 개를 반환해야 모델이 받아서 학습할 수 있음
        return image, label

''' 모든 데이터 관리 로직(경로 설정, 데이터 분할, 로더 생성) '''
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, num_workers):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 이미지 크기 조절, 정규화, 텐서 변환을 위한 기본 변환기
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        df = pd.read_csv(f"{self.path}/train.csv")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        train_idx, val_idx = next(iter(skf.split(df, df['target'])))
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 만들어 놓은 CustomDataset 클래스를 사용.
        image_data_root = f"{self.path}/train"
        self.train_dataset = CustomDataset(df=train_df, data_root=image_data_root, transform=self.transform)
        self.val_dataset = CustomDataset(df=val_df, data_root=image_data_root, transform=self.transform)

        print(f"Train/Val 데이터셋 분리 및 생성 완료. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        # DataLoader를 반환.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        # DataLoader를 반환.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )