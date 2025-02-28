from MedsegDiff_V1_Model.model import MedSegDiffV1
from torch.utils.data import DataLoader
from eval import evaluate_model
import torch

# Load Model
model_path = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedSegDiffV1(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path))  # load weight

# Load Dataset
#test_dataset = ProstateDataset("test")  # change to the dataloader script
#test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Run Evaluate
#evaluate_model(model, test_loader, device)
