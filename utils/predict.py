import re
from utils.dataloader import MyDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def get_test():
    test = []
    with open ("/kaggle/input/lab5-data/test_without_label.txt") as f:
        f.readline()
        for line in f:
            test.append(line.strip())
    return test
        
def cleanText(text):
        text = re.sub('@[A-Za-z0-9_]+', '', text) 
        text = re.sub('[^a-zA-Z]', ' ', str(text).lower().strip())
        text = re.sub('#','',text) 
        text = re.sub('RT[\s]+','',text)
        text = re.sub('https?:\/\/\S+', '', text) 
        return text
    
def predict(model, text_processor, img_processor):
    label_decoder = {"0": "positive", "1": "negative", "2": "neutral"}
    pre_dict = {}
    model = model.to('cpu')
    test = get_test()
    labelEncoder = {"positive": 0, "negative": 1, "neutral": 2, "null": 3}
    test_dataset = MyDataset("/kaggle/input/lab5-data/data", test, labelEncoder, text_processor, img_processor)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    for idx, img, input_ids, attention_mask, token_type_ids, labels in tqdm(test_dataloader):
        text_input = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        with torch.no_grad():
            output = model(text_input, img)
            labels = output.max(1)[1]
            for i, label in zip(idx, labels):
                pre_dict[i] = label_decoder[str(label.item())]
    return pre_dict
