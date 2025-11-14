import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
from tqdm import tqdm

from transformers import PatchTSTConfig, PatchTSTForRegression

from data_utils.feature_engineering import build_patchtst_dataset
from utils.logger import get_logger

logger = get_logger("train_patchtst")

DATA_PATH = "data/train_history.csv"
MODEL_DIR = "models/patchtst_cap"  # 디렉토리로 저장 (HF 방식)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    logger.info("PatchTST 학습용 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_patchtst_dataset(df)
    logger.info(f"X shape={X.shape}, y shape={y.shape}")  # (N, seq_len, num_channels)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    seq_len = X.shape[1]
    num_channels = X.shape[2]

    # Hugging Face PatchTST 설정
    config = PatchTSTConfig(
        num_input_channels=num_channels,  # 4개: steps, avg_hr, work_hours, deliveries
        context_length=seq_len,          # 시퀀스 길이 (예: 7)
        num_targets=1,                   # capacity 1개
        prediction_length=1,             # 회귀지만 1로 맞춰둠
        loss="mse",                      # 회귀이므로 mse
        patch_length=1,
        patch_stride=1,
    )

    model = PatchTSTForRegression(config=config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger.info(f"학습 시작 (device={DEVICE}) epochs={epochs}")
    for epoch in range(1, epochs + 1):
        # ---------------- train ----------------
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            # HF PatchTST: past_values, target_values 넣으면 loss 계산해줌
            outputs = model(past_values=xb, target_values=yb)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------------- val ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(past_values=xb, target_values=yb)
                loss = outputs.loss
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        logger.info(f"Epoch {epoch}: train MSE={train_loss:.3f}, val MSE={val_loss:.3f}")

    # Hugging Face 방식으로 디렉토리에 저장
    model.save_pretrained(MODEL_DIR)
    logger.info(f"✅ PatchTST 모델 저장 완료: {MODEL_DIR}")


if __name__ == "__main__":
    main()
