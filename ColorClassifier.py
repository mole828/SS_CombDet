from pathlib import Path
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

## do not change colors
color_index = ['white', 'yellow', 'green', 'pink', 'blue', 'purple', 'red', 'black', 'unknown']

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, len(color_index))  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image_paths: list[Path]
    labels: list[int]
    def __init__(self, image_forder: Path):
        if not image_forder.is_dir():
            raise TypeError("image_forder is not a directory.")
        self.image_paths = []
        self.labels = []
        for file in image_forder.iterdir():
            if file.name.endswith('.jpg'):
                words = file.name.split('_')
                colors = [words[-1].split('.')[0], words[-3], words[-4].split('#')[0]]
                if len(set(colors)) != 1:
                    raise ValueError(f"diff color in file name, file: {file.name}")
                color = colors[0]
                self.image_paths.append(file)
                self.labels.append(color_index.index(color))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class ColorClassifier:
    device: torch.device
    net: SimpleCNN
    transform:Callable[[Image.Image], torch.Tensor]= CustomDataset.transform
    def __init__(self, model_path: str='color_classifier.pt') -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = SimpleCNN().to(self.device)
        if Path(model_path).exists(): 
            state_dict = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(state_dict=state_dict)
        else:
            raise FileNotFoundError('model not found')
    
    def classify(self, image: Image.Image) -> str:
        input = self.transform(image).unsqueeze(0).to(self.device)
        output = self.net(input)
        _, predicted = torch.max(output, 1)
        index:int = predicted.item() 
        return color_index[index]
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    if Path('color_classifier.pt').exists():
        state_dict = torch.load('color_classifier.pt')
        net.load_state_dict(state_dict=state_dict)

    dataset = CustomDataset(Path('/home/mole/projects/python/yolo/SS_CombDet/datasets/crops'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    try:
    # 训练网络
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    except KeyboardInterrupt:
        print('stop train')
    finally:
        torch.save(net.state_dict(), Path('./color_classifier.pt'))

    print("Finished Training")


    def classify_color(image_path, model, transform, device):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)  # 移动数据到同样的设备
        output = model(image)
        print(output)
        _, predicted = torch.max(output, 1)
        return predicted.item()

    # 示例：
    new_image_path = "/home/mole/projects/python/yolo/balls/crops/0621200001_K13_6492daf31caf140050759937_15_36568b_blue#1_blue_25_blue.jpg"
    predicted_color = classify_color(new_image_path, net, CustomDataset.transform, device)
    print(f"Predicted color category: {color_index[predicted_color]}")