import torch
from torch.utils.data import Dataset

class SUMDataset(Dataset):
    def __init__(self, split, data):
        self.split = split
        self.input_ids = data['model_inputs']['input_ids']
        self.attention_mask = data['model_inputs']['attention_mask']
        self.decoder_input_ids = data['labels']['input_ids']
        self.decoder_attention_mask = data['labels']['attention_mask']
        self.ids = data['ids']
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_mask[idx]), torch.tensor(self.decoder_input_ids[idx]), torch.tensor(self.decoder_attention_mask[idx]), self.ids[idx]
