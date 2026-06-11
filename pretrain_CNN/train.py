import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

from dataset import SuperResolutionDataset
from Simple_CNN import SimpleCNN
from loss import image_compare_loss

torch.backends.cudnn.enabled = False

# trian function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for outputs in tqdm(train_loader):
        inputs = outputs[0]
        targets = outputs[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


# evaluate function
def calculate_psnr_batch(pred_batch, target_batch, max_val=1.0):
    """Calculate the mean PSNR of all images within a batch"""
    batch_size = pred_batch.size(0)
    mse = torch.mean(torch.pow(pred_batch - target_batch, 2), dim=[1, 2, 3])
    psnr = torch.zeros(batch_size, dtype=torch.float32, device=pred_batch.device)
    psnr[mse > 0] = 10 * torch.log10((max_val ** 2) / mse[mse > 0])
    avg_psnr = torch.mean(psnr)
    return avg_psnr.item()


def evaluate(model, dataloader, device):
    root = './dataset/FFHQ_32_128/cnn_sr_32_128'
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Test', leave=False):
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_psnr += calculate_psnr_batch(outputs, targets)
    avg_psnr = total_psnr / len(dataloader)
    return avg_psnr

# Save the CNN prediction results
def save_res(model, dataloader, device):
    root = './dataset/FFHQ_32_128'
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Test', leave=False):
            inputs, targets , path , _ = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            image = ((outputs[0] + 1) / 2)
            path = path[0].replace('lr_32','cnn_sr_32_128')
            save_image(image, path)


def main():
    # Setting parameters
    scale_factor = 4
    batch_size = 128
    lr = 1e-4
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hr_dir = './dataset/FFHQ_32_128/train/hr_128'
    lr_dir = './dataset/FFHQ_32_128/train/lr_32'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset
    train_dataset = SuperResolutionDataset(hr_dir, lr_dir, transform)
    test_dataset = SuperResolutionDataset(hr_dir, lr_dir, transform,train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create models, loss functions and optimizers
    model = SimpleCNN(scale_factor=scale_factor).to(device)
    # model.load_state_dict(torch.load('pretrain_CNN/cnn_weights.pth'))
    criterion = image_compare_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and evaluation models, and save the model weights along with prediction images
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_psnr = evaluate(model, test_loader, device)
        print('Epoch [{}/{}], Train Loss: {:.4f}, '
              'Test PSNR: {:.4f}'.format(epoch + 1, epochs, train_loss, test_psnr))

        torch.save(model.state_dict(), 'cnn_weights_not_res.pth')

    save_res(model, test_loader, device)


if __name__ == '__main__':
    main()
