import torch.nn as nn 
import torch
from torchvision import models
from transformers import ViTModel, BertModel, BertForSequenceClassification
from torchvision.models.resnet import ResNet50_Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResFeatModel(nn.Module):
    def __init__(self):
        super(ResFeatModel, self).__init__()

        self.cnn = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1000, 512)

    def forward(self, img):
        Y = self.cnn(img)
        Y = self.fc(Y)
        return Y
    
class BertFeatModel(nn.Module):
    def __init__(self):
        super(BertFeatModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for n, param in self.bert.named_parameters():
            if not n.startswith("pooler.dense"):
                param.requires_grad = False
        self.fc_bert = nn.Linear(768, 512)
        self.norm = nn.LayerNorm(512)
        
    def forward(self, text):
        text_feat = self.bert(**text)
        text_feat = self.fc_bert(text_feat[0].mean(1))
        text_feat = self.norm(text_feat)
        return text_feat

class VitFeatModel(nn.Module):
    def __init__(self):
        super(VitFeatModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for n, param in self.vit.named_parameters():
            if not n.startswith("pooler.dense"):
                param.requires_grad = False
        self.fc = nn.Linear(768, 512)
        self.norm = nn.LayerNorm(512)
        
    def forward(self, img):
        Y = self.vit(img)
        feat = Y.last_hidden_state.mean(1)
        feat = self.fc(feat)
        feat = self.norm(feat)
        return feat
    

