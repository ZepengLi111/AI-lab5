import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import ViTModel, BertModel, BertForSequenceClassification
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.attention1 = nn.MultiheadAttention(512, 8)
        self.attention2 = nn.MultiheadAttention(512, 8)
    def forward(self, text_feat, img_feat):
        text_feat = self.attention1(text_feat, img_feat, img_feat)[0]
        image_feat = self.attention2(img_feat, text_feat, text_feat)[0]
        return text_feat, img_feat
    
class AttentionModel(nn.Module):
    def __init__(self, text_model, img_model):
        super(AttentionModel, self).__init__()
        self.text_feat_model = text_model.to(device)
        self.img_feat_model = img_model.to(device)
        self.attentionLayer1 = AttentionLayer().to(device)
        self.attentionLayer2 = AttentionLayer().to(device)
        self.attentionLayer3 = AttentionLayer().to(device)
        self.final_fc = nn.Linear(1024, 3)
        
    def forward(self, text, img):
        text_feat = self.text_feat_model(text)
        img_feat = self.img_feat_model(img)
        text_feat, img_feat = self.attentionLayer1(text_feat, img_feat)
        text_feat, img_feat = self.attentionLayer2(text_feat, img_feat)
        text_feat, img_feat = self.attentionLayer3(text_feat, img_feat)
        # 拼接特征    
        final_feat = torch.cat([text_feat, img_feat], dim=-1)
        output = self.final_fc(final_feat)
        return output
    
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.img_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = torch.nn.Linear(768, 256)
        self.image_proj = torch.nn.Linear(768, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        
        self.fusion_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4), 2)
        self.fc = nn.Linear(256, 3)
    
    def forward(self, text, img):
        text_encoding = self.text_model(**text)[0]
        img_encoding = self.img_model(img)[0]
        text_encoding = self.norm1(self.text_proj(text_encoding))
        img_encoding = self.norm2(self.image_proj(img_encoding))
        concat_encoding = torch.cat([text_encoding, img_encoding], dim=1)
        fusion_encoding = self.fusion_model(concat_encoding)
        cls_encoding = fusion_encoding[:, 0, :]
        out = self.fc(cls_encoding)
        return out

class TransformerModel2(nn.Module):
    def __init__(self):
        super(TransformerModel2, self).__init__()
        self.img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = torch.nn.Linear(768, 256)
        self.image_proj = torch.nn.Linear(1000, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        
        self.fusion_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 4), 2)
        self.fc = nn.Linear(512, 3)
    
    def forward(self, text, img):
        text_encoding = self.text_model(**text)[1]
        img_encoding = self.img_model(img)
        text_encoding = self.norm1(self.text_proj(text_encoding))
        img_encoding = self.norm2(self.image_proj(img_encoding))
        concat_encoding = torch.cat([text_encoding, img_encoding], dim=1)
        fusion_encoding = self.fusion_model(concat_encoding)
        cls_encoding = fusion_encoding
        out = self.fc(cls_encoding)
        return out
