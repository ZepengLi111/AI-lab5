import torch
from tqdm import tqdm

def evaluate(model, val_dataloader, loss_fn, device, ablation=None, tokenizer=None):
    model.to(device)
    total_correct = 0
    total_n_samples = 0
    model.eval()
    if ablation == 'img':
            blank_tokens = tokenizer("", max_length=64, padding='max_length', truncation=True, return_tensors='pt')
    val_loss = 0
    with torch.no_grad():
        for idx, img, input_ids, attention_mask, token_type_ids, labels in val_dataloader:
            if ablation == 'text':
                img_input = torch.zeros(img.size()).to(device)
            else:
                img_input = img.to(device)
            if ablation == 'img':
                text_input = {'input_ids': blank_tokens['input_ids'].repeat(64, 1).to(device), 'attention_mask': blank_tokens['attention_mask'].repeat(64, 1).to(device), 'token_type_ids': blank_tokens['token_type_ids'].repeat(64, 1).to(device)}
            else:
                text_input = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'token_type_ids': token_type_ids.to(device)}
                
            labels = labels.to(device)
            outputs = model(text_input, img_input)
#             outputs = model(img)
            prediction = outputs.argmax(dim=1)
            l = loss_fn(outputs, labels)
            val_loss += l.item()
            total_correct += (prediction == labels).sum().item()
            total_n_samples += len(img)
    accuracy = total_correct / total_n_samples
    val_loss /= total_n_samples
    return accuracy, val_loss

def train(model, epoch, optimizer, loss_func, train_dataloader, val_dataloader, device, ablation=None, tokenizer=None, patience=3):
    best_loss = float('inf')
    best_epoch = 0
    best_acc = 0
    for e in range(epoch):
        loss_sum = 0
        if ablation == 'img':
            blank_tokens = tokenizer("", max_length=64, padding='max_length', truncation=True, return_tensors='pt')
        for idx, img, input_ids, attention_mask, token_type_ids, labels in tqdm(train_dataloader):
            if ablation == 'text':
                img_input = torch.zeros(img.size()).to(device)
            else:
                img_input = img.to(device)
            if ablation == 'img':
                text_input = {'input_ids': blank_tokens['input_ids'].repeat(64, 1).to(device), 'attention_mask': blank_tokens['attention_mask'].repeat(64, 1).to(device), 'token_type_ids': blank_tokens['token_type_ids'].repeat(64, 1).to(device)}
            else:
                text_input = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'token_type_ids': token_type_ids.to(device)}
                
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(text_input, img_input)
            loss = loss_func(output, labels)
            loss.backward() 
            optimizer.step()
            loss_sum += loss
        val_acc, val_loss = evaluate(model, val_dataloader, loss_func, device)
        if val_loss < best_loss:
            torch.save(model.state_dict(), 'model_loss.pth') 
            best_loss = val_loss
            best_epoch = epoch
            if e - best_epoch > patience:
                break
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'model_acc.pth')
            best_acc = val_acc
        print("epoch: ", e + 1)
        print("val", val_acc, val_loss)
        print("train", loss_sum.item())
    print("best epoch: ", best_epoch)
    print("best loss: ", best_loss)
    print("best acc: ", best_acc)
    return best_acc
