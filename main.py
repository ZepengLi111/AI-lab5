from utils.dataloader import get_dataloader, read_labels
from utils.train import train
from model.VL_model import AttentionModel, TransformerModel, TransformerModel2
from model.encoder import VitFeatModel, BertFeatModel, ResFeatModel
from transformers import BertTokenizer
import torch.nn as nn
from torchvision import transforms
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--model', default=None, type=str)      #['1', '2', '3', '4']
parser.add_argument('--ablation', default=None, type=str)  #['img', 'text']
parser.add_argument('--epoch', default=10, type=int)


def main():
    args = parser.parse_args()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
])
    train_labels = read_labels()

    train_dataloader, val_dataloader = get_dataloader(bert_tokenizer, img_transform, train_labels, batch_size=args.batch_size)
    if args.model == '1':
        # resnet + bert + transformer
        model = TransformerModel2().to(device)
    elif args.model == '2':
        # vit + bert + transformer
        model = TransformerModel().to(device)
    elif args.model == '3':
        # resnet + bert + attention
        model = AttentionModel(BertFeatModel(), ResFeatModel()).to(device)
    elif args.model == '4':
        # vit + bert + attention
        model = AttentionModel(BertFeatModel(), VitFeatModel()).to(device)
    else:
        raise ValueError('model should be in [1, 2, 3, 4]')
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_epoch = args.epoch
    train(model, n_epoch, optimizer, loss, train_dataloader, val_dataloader, device=device, ablation=args.ablation, tokenizer=bert_tokenizer)

if __name__ == "__main__":
    main()
