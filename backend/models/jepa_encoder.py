import torch 
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


class JEPAEncoder:
    def __init__(self, model_name="vit_base_patch16_224"):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Vision Transformer backbone
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0   # remove classifier
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.model.num_features, 768),
            nn.ReLU(),
            nn.Linear(768,512)
        ).to(self.device)

        print("JEPA Encoder Ready.")

    def encode_image(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            features = self.model(image)

            embedding = self.projector(features)

            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()
        
