import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from engine import train
from vit import ViT
from utils import set_seeds
from data_setup import load_cifar10

def main(args):
    
    # Seed for reproducibility
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Creating Data Loaders
    train_dataloader, test_dataloader, classes = load_cifar10(args.batch_size)

    # Initializing model variants as per choice
    if args.model == "base":
        vit = ViT(
            img_size =  32,
            num_transformer_layers = 12,
            embedding_dim = 768,
            mlp_size = 3072,
            num_heads = 12,
            num_classes = len(classes)
        )
    if args.model == "large":
        vit = ViT(
            img_size =  32,
            num_transformer_layers = 24,
            embedding_dim = 1024,
            mlp_size = 4096,
            num_heads = 16,
            num_classes = len(classes)
        )
    if args.model == "huge":
        vit = ViT(
            img_size =  32,
            num_transformer_layers = 32,
            embedding_dim = 1280,
            mlp_size = 5120,
            num_heads = 16,
            num_classes = len(classes)
        )       

    # Adam Optimizer with parameters from the ViT paper 
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()


    # Train the model and save the training results to a dictionary
    results = train(model=vit,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.num_epochs,
                        device=device)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer (ViT) model")
    parser.add_argument("--model", type=str, default="base", help="ViT Variant to train")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    main(args)
