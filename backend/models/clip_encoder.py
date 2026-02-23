import torch
import open_clip
from PIL import Image

class CLIPEncoder:
    def __init__(self,model_name='ViT-B-32', pretrained='openai'):
        print("Loading CLIP model")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        #GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"CLIP model loaded on {self.device}")
        
    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)

        # Normalize embedding
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding.cpu().numpy()
    
    def encode_text(self, text:str):
        text_tokens = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            
            #Normalize embedding
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            return text_embedding.cpu().numpy()
        
    def encoded_pair(self, image_path: str, caption: str):
        img_emb = self.encode_image(image_path)
        text_emb = self.encode_text(caption)
        return img_emb, text_emb

        