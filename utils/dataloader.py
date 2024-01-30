import re
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torch.utils.data import random_split

def read_labels():
    labels = []
    with open("./data/train.txt", "r") as f:
        for line in f:
            labels.append(line.strip())
    return labels

class MyDataset(Dataset):
    def __init__(self, root_dir, labels, label_encoder, tokenizer, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.labels)
    
    def cleanText(self, text):
        text = re.sub('@[A-Za-z0-9_]+', '', text) 
        text = re.sub('[^a-zA-Z]', ' ', str(text).lower().strip())
        text = re.sub('#','',text) 
        text = re.sub('RT[\s]+','',text)
        text = re.sub('https?:\/\/\S+', '', text) 
        return text

    def __getitem__(self, idx):
        index = self.labels[idx].split(",")[0]
        img_path = os.path.join(self.root_dir, f"{index}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx].split(",")[1]
        label = self.label_encoder[label]
        text_path = os.path.join(self.root_dir, f"{index}.txt")
        with open(text_path, 'r', encoding='latin-1') as f:
            text = f.readline()
        text = self.cleanText(text)
        tokens = self.tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
        if self.transform:
            image = self.transform(image)
#         print(type(image))
#         if type(image) == dict:
#             image = image['pixel_values']

        return index, image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), tokens['token_type_ids'].squeeze(), label
    
def get_dataloader(text_processor, img_processor, labels, batch_size=64):
    labelEncoder = {"positive": 0, "negative": 1, "neutral": 2, "null": 3}
    num_val = int(len(labels) * 0.2)
    num_train = len(labels) - num_val
    train_data, val_data = random_split(labels, [num_train, num_val], generator=torch.Generator().manual_seed(42))
    
    train_dataset = MyDataset("./data/data/", train_data, labelEncoder, text_processor, img_processor)
    val_dataset = MyDataset("./data/data/", val_data, labelEncoder, text_processor, img_processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader
