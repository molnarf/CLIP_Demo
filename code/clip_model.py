from cgitb import text
import clip
import torch
import numpy as np

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.eval()


def preprocess_images(images: list) -> list:
    preprocessed_images = [preprocess(image) for image in images]
    return preprocessed_images


def run_clip(labels: list[str], images: list):
    '''
    Generate image and text embeddings and return their cosine similarity.

    Arguments:
        labels: A list of str labels (size N)
        images: A list of PIL images (size M)
    
    Returns:
        similarity: A numpy matrix containing the cosine similarities of shape NxM
    '''
    images = preprocess_images(images)
    image_input = torch.tensor(np.stack(images))
    text_tokens = clip.tokenize(["A photo of a " + label for label in labels])

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()    # shape: [#imgs, 512]
        text_features = model.encode_text(text_tokens).float()      # shape: [#labels, 512]

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = image_features @ text_features.T

    return similarity

def get_n_best_matches(labels: list, images: list, n: int = 3):
    n = min(n, len(labels))
    similarities = torch.tensor(run_clip(labels, images))

    text_probs = (100.0 * similarities).softmax(dim=-1)
    top_probs, top_labels = text_probs.topk(n, dim=-1)
    return top_probs, top_labels

