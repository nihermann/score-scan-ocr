from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from model import TinyConvNet  # assumes model code is in model.py

split_inference_models = {}

def load_model(model_name: str = "tinyv2.1", device: str = 'cuda' if torch.cuda.is_available() else "cpu") -> TinyConvNet:
    """
    Load a pre-trained model from disk.
    :param model_name: str - name of the model to load (e.g., 'tinyv1'). It's expected that the model file is named 'models/model_<model_name>.pth'.
    :param device: str - device to load the model on ('cuda' or 'cpu')
    :return: TinyConvNet - the loaded model
    """
    model_path = Path("models") / ("model_" + model_name + ".pth")
    model = TinyConvNet(input_channels=1, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def predict_tensor(tensor: Union[torch.Tensor, np.ndarray], model_name: str = 'tinyv2.1', device='cuda' if torch.cuda.is_available() else "cpu") -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Predict the class and confidence of a tensor using a pre-trained model.
    :param tensor: torch.Tensor or np.ndarray - the input tensor to predict. It should be of shape [B, 1, H, W] where B is batch size, 1 is the number of channels (grayscale), and H, W are height and width.
    :param model_name: str - the name of the model to use for prediction. It should match the model file in 'models/model_<model_name>.pth'.
    :param device: str - the device to run the model on ('cuda' or 'cpu'). Default is 'cuda' if available, otherwise 'cpu'.
    :return: Tuple - a tuple containing the predicted class and confidence score. If the input tensor is a single sample, it returns (predicted_class, confidence_score) as floats. If the input is a batch, it returns (predicted_classes, confidence_scores) as numpy arrays or tensors.
    """
    if isnp := isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if model_name not in split_inference_models:
        split_inference_models[model_name] = load_model(model_name, device)
    model = split_inference_models[model_name]
    model.eval()
    while tensor.ndim < 4:  # Ensure tensor has 4 dimensions: [B, 1, H, W]
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)  # shape: [B, 1, H, W]
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
    if tensor.shape[0] == 1:
        return pred.item(), confidence.item()
    if isnp:
        return pred.cpu().numpy(), confidence.cpu().numpy()
    return pred, confidence


if __name__ == "__main__":
    from data_loader import DownsampledTensorDataset

    # Example: predict the first sample
    dataset = DownsampledTensorDataset("down_data")
    mapping = ["page", "title"]
    # tensor, label = dataset[0]

    tensors, labels = zip(*[dataset[i] for i in range(-20, 20)])  # Get first 10 samples
    tensors = torch.stack(tensors)  # shape: [10, 1, H, W]
    labels = torch.tensor(labels)  # shape: [10]

    v = "tinyv1"
    pred, conf = predict_tensor(tensors, v)

    for i in range(40):
        print(f"Predicted: {mapping[pred[i].item()]} (Confidence: {conf[i].item():.2%}), Ground Truth: {mapping[labels[i].item()]}")
