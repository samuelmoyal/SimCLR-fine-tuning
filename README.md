# SimCLR-fine-tuning
Fine-tuning of SimCLR for one-shot learning of plan symbols

This code provides a pipeline to fine-tune SimCLR with any plan dataset with bounding box on symbols. 

Cropping:
- Crop the symbols thanks to the bounding boxes

Training (unsupervised):
- dataset of unlabeled images
- for each image: generation of two augmented views using strong transforms (random crop-resize, color jitter, horizontal flip). A batch of N images becomes 2N views
- model optimize the NT-Xent contrastive loss, to increase similarity between the two views of the same image (positive pair) and pushes apart all other images in the batch (negative pairs)


Architecture:
- ResNet_backbone + Projection head(Linear + ReLU + Linear)
- Head removed for inference

Inference (one example per class):
- The user provides one reference image per class.
- New image encoded with the backbone and classified using nearest-neighbor with cosine similarity in the embedding space.


Reference: SimCLR (Chen et al., 2020) — https://arxiv.org/abs/2002.05709
