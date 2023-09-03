import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from model.model_with_prompt import PromptModel
from datasets import RandomDataset

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val
        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image


test_data = RandomDataset()

database_loader = torch.utils.data.DataLoader(test_data, batch_size=3, shuffle=False, )

model = PromptModel()

text=["a photo of a cat","an image of a dog", "a image of a moderately differentiated cancer"]

for i, img in enumerate(database_loader):
    features = model(img, text)
    
    print(features.shape)
