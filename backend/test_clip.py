from models.clip_encoder import CLIPEncoder
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b.T)


if __name__ == "__main__":
    encoder = CLIPEncoder()

    # Example test
    image_path = "backend/data/images/submarine.jpg"
    caption = "A modern animated submarine floating underwater with on inside and has bubble stream coming out of it"

    img_emb = encoder.encode_image(image_path)
    txt_emb = encoder.encode_text(caption)

    print("Image Embedding Shape:", img_emb.shape)
    print("Text Embedding Shape:", txt_emb.shape)

    sim = cosine_similarity(img_emb, txt_emb)
    print("Cosine Similarity:", sim[0][0])
