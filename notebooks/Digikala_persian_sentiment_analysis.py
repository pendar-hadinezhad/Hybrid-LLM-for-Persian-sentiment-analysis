# ============================================================
# DIGIKALA PERSIAN SENTIMENT ANALYSIS
# ParsBERT + Statistical/Behavioral Features
# Simple Fusion + Attention Fusion model
# ============================================================

import os
import re
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "digikala.csv"

PROCESSED_FILE = "digikala_processed.csv"

MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"

RANDOM_STATE = 42

# برای تست اولیه
# مثلاً 100000 قرار بده.
# برای اجرای کامل:
# MAX_ROWS = None
MAX_ROWS = 100000

BATCH_SIZE = 32

MAX_LENGTH = 128

NUM_EPOCHS = 5

LEARNING_RATE = 2e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# 3. LOAD + PREPROCESS DATA IN CHUNKS
# ============================================================

USE_COLS = [
    "id",
    "title",
    "body",
    "created_at",
    "rate",
    "is_buyer",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes"
]


def clean_persian_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # حذف URL
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    # حذف HTML
    text = re.sub(
        r'<[^>]+>',
        ' ',
        text
    )

    # تبدیل فاصله‌های متعدد
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


def process_chunk(chunk):

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_columns = [
        "title",
        "body",
        "advantages",
        "disadvantages"
    ]

    for col in text_columns:

        chunk[col] = (
            chunk[col]
            .fillna("")
            .astype(str)
            .apply(clean_persian_text)
        )

    # ترکیب متن‌ها
    chunk["text"] = (
        chunk["title"] + " " +
        chunk["body"] + " " +
        chunk["advantages"] + " " +
        chunk["disadvantages"]
    )

    # حذف متن‌های خالی
    chunk = chunk[
        chunk["text"].str.len() > 0
    ].copy()


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    chunk["rate"] = pd.to_numeric(
        chunk["rate"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["rate"]
    )


    # sentiment labels

    def convert_rating(rate):

        if rate <= 2:
            return 0       # Negative

        elif rate == 3:
            return 1       # Neutral

        else:
            return 2       # Positive


    chunk["label"] = (
        chunk["rate"]
        .apply(convert_rating)
        .astype(int)
    )


    # --------------------------------------------------------
    # BEHAVIORAL FEATURES
    # --------------------------------------------------------

    for col in [
        "likes",
        "dislikes",
        "is_buyer"
    ]:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    chunk["created_at"] = pd.to_datetime(
        chunk["created_at"],
        errors="coerce"
    )

    chunk["hour"] = (
        chunk["created_at"]
        .dt.hour
        .fillna(0)
    )

    chunk["day_of_week"] = (
        chunk["created_at"]
        .dt.dayofweek
        .fillna(0)
    )


    # --------------------------------------------------------
    # FINAL FEATURES
    # --------------------------------------------------------

    final_columns = [
        "id",
        "text",
        "likes",
        "dislikes",
        "is_buyer",
        "hour",
        "day_of_week",
        "label"
    ]

    return chunk[final_columns]


def preprocess_large_csv():

    first_chunk = True

    total_rows = 0

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=USE_COLS,
        chunksize=100_000,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader):

        print(
            f"\nProcessing chunk {chunk_number}"
        )

        processed = process_chunk(chunk)

        if MAX_ROWS is not None:

            remaining = (
                MAX_ROWS - total_rows
            )

            if remaining <= 0:
                break

            processed = processed.iloc[
                :remaining
            ]

        processed.to_csv(
            PROCESSED_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8"
        )

        total_rows += len(processed)

        print(
            "Processed rows:",
            total_rows
        )

        first_chunk = False

        if (
            MAX_ROWS is not None
            and total_rows >= MAX_ROWS
        ):
            break


    print(
        "\nPreprocessing completed."
    )

    print(
        "Total rows:",
        total_rows
    )


# ============================================================
# 4. RUN PREPROCESSING
# ============================================================

if not os.path.exists(PROCESSED_FILE):

    preprocess_large_csv()

else:

    print(
        "Processed file already exists."
    )


# ============================================================
# 5. LOAD PROCESSED DATA
# ============================================================

df = pd.read_csv(
    PROCESSED_FILE
)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(
    df["label"].value_counts(
        normalize=True
    )
)


# ============================================================
# 6. REMOVE DUPLICATE TEXT
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

after = len(df)

print(
    f"\nRemoved {before - after} duplicate comments."
)


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)

print("\nDataset sizes:")

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 8. STATISTICAL / BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    "likes",
    "dislikes",
    "is_buyer",
    "hour",
    "day_of_week"
]


scaler = StandardScaler()

train_df[
    BEHAVIORAL_FEATURES
] = scaler.fit_transform(
    train_df[BEHAVIORAL_FEATURES]
)

val_df[
    BEHAVIORAL_FEATURES
] = scaler.transform(
    val_df[BEHAVIORAL_FEATURES]
)

test_df[
    BEHAVIORAL_FEATURES
] = scaler.transform(
    test_df[BEHAVIORAL_FEATURES]
)


# ============================================================
# 9. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 10. DATASET
# ============================================================

class DigikalaDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.texts = (
            dataframe["text"]
            .tolist()
        )

        self.behavioral = (
            dataframe[
                BEHAVIORAL_FEATURES
            ]
            .values
            .astype(np.float32)
        )

        self.labels = (
            dataframe["label"]
            .values
            .astype(np.int64)
        )


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        text = self.texts[index]

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {

            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

            "behavioral":
                torch.tensor(
                    self.behavioral[index],
                    dtype=torch.float
                ),

            "label":
                torch.tensor(
                    self.labels[index],
                    dtype=torch.long
                )
        }

        return item


# ============================================================
# 11. DATALOADERS
# ============================================================

train_dataset = DigikalaDataset(
    train_df
)

val_dataset = DigikalaDataset(
    val_df
)

test_dataset = DigikalaDataset(
    test_df
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 12. PARSBERT ENCODER
# ============================================================

class ParsBERTEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.hidden_size = (
            self.bert.config.hidden_size
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS representation

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        return cls_embedding


# ============================================================
# 13. SIMPLE FUSION MODEL
# ============================================================

class SimpleFusionModel(nn.Module):

    def __init__(
        self,
        text_dim=768,
        behavioral_dim=5,
        num_classes=3
    ):

        super().__init__()

        self.text_encoder = (
            ParsBERTEncoder()
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                text_dim + behavioral_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        behavioral
    ):

        text_embedding = (
            self.text_encoder(
                input_ids,
                attention_mask
            )
        )

        combined = torch.cat(
            [
                text_embedding,
                behavioral
            ],
            dim=1
        )

        logits = self.classifier(
            combined
        )

        return logits


# ============================================================
# 14. ATTENTION FUSION MODEL
# ============================================================

class AttentionFusionModel(nn.Module):

    def __init__(
        self,
        text_dim=768,
        behavioral_dim=5,
        num_classes=3
    ):

        super().__init__()

        self.text_encoder = (
            ParsBERTEncoder()
        )

        # تبدیل behavioral features
        # به فضای embedding

        self.behavior_projection = nn.Sequential(

            nn.Linear(
                behavioral_dim,
                text_dim
            ),

            nn.ReLU()
        )


        # Attention score

        self.attention = nn.Sequential(

            nn.Linear(
                text_dim * 2,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                2
            )
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                text_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        behavioral
    ):

        # ----------------------------------------------------
        # Text representation
        # ----------------------------------------------------

        text_embedding = (
            self.text_encoder(
                input_ids,
                attention_mask
            )
        )


        # ----------------------------------------------------
        # Behavioral representation
        # ----------------------------------------------------

        behavioral_embedding = (
            self.behavior_projection(
                behavioral
            )
        )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        combined_for_attention = torch.cat(
            [
                text_embedding,
                behavioral_embedding
            ],
            dim=1
        )


        attention_scores = (
            self.attention(
                combined_for_attention
            )
        )


        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )


        text_weight = (
            attention_weights[:, 0:1]
        )

        behavioral_weight = (
            attention_weights[:, 1:2]
        )


        # ----------------------------------------------------
        # Weighted Fusion
        # ----------------------------------------------------

        fused = (
            text_weight * text_embedding
            +
            behavioral_weight *
            behavioral_embedding
        )


        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# 15. TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    for epoch in range(NUM_EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}"
        ):

            input_ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            behavioral = (
                batch["behavioral"]
                .to(DEVICE)
            )

            labels = (
                batch["label"]
                .to(DEVICE)
            )


            optimizer.zero_grad()


            logits = model(
                input_ids,
                attention_mask,
                behavioral
            )


            loss = criterion(
                logits,
                labels
            )


            loss.backward()

            optimizer.step()


            train_loss += (
                loss.item()
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_f1 = evaluate_model(
            model,
            val_loader
        )


        print(
            f"\nEpoch {epoch + 1}"
        )

        print(
            "Train Loss:",
            train_loss / len(train_loader)
        )

        print(
            "Validation Macro-F1:",
            val_f1
        )


# ============================================================
# 16. EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    predictions = []

    labels_all = []


    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            behavioral = (
                batch["behavioral"]
                .to(DEVICE)
            )

            labels = (
                batch["label"]
                .to(DEVICE)
            )


            logits = model(
                input_ids,
                attention_mask,
                behavioral
            )


            predictions.extend(
                torch.argmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )


    accuracy = accuracy_score(
        labels_all,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels_all,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    print("\nAccuracy:", accuracy)

    print(
        "Macro Precision:",
        precision
    )

    print(
        "Macro Recall:",
        recall
    )

    print(
        "Macro F1:",
        f1
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            labels_all,
            predictions,
            target_names=[
                "Negative",
                "Neutral",
                "Positive"
            ],
            zero_division=0
        )
    )


    return f1


# ============================================================
# 17. SIMPLE FUSION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING SIMPLE FUSION MODEL")
print("=" * 60)


simple_model = SimpleFusionModel(
    text_dim=768,
    behavioral_dim=len(
        BEHAVIORAL_FEATURES
    ),
    num_classes=3
)


train_model(
    simple_model,
    train_loader,
    val_loader
)


print("\nFINAL TEST - SIMPLE FUSION")

evaluate_model(
    simple_model,
    test_loader
)


# ============================================================
# 18. ATTENTION FUSION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING ATTENTION FUSION MODEL")
print("=" * 60)


attention_model = AttentionFusionModel(
    text_dim=768,
    behavioral_dim=len(
        BEHAVIORAL_FEATURES
    ),
    num_classes=3
)


train_model(
    attention_model,
    train_loader,
    val_loader
)


print("\nFINAL TEST - ATTENTION FUSION")

evaluate_model(
    attention_model,
    test_loader
)