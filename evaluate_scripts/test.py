from MedsegDiff_V1_Model.model import MedSegDiffV1
from MedsegDiff_V1_Model.custom_dataset_loader import CustomDataset
from eval import evaluate_model
import torch
import torchvision.transforms as transforms
# Load Model
model_path = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedSegDiffV1(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path))  # load weight

# Load Dataset
data_path = ""
tran_list = [transforms.Resize((256,256)), transforms.ToTensor()]
transform_test = transforms.Compose(tran_list)

ds = CustomDataset(data_path, transform_test, mode = 'Test')
#in_ch = 4

datal = torch.utils.data.DataLoader(
        ds,
        batch_size=16,
        shuffle=True)
data = iter(datal)

# Run Evaluate
evaluate_model(model, data, device)
