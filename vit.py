import torch 
import torch.nn as nn
from torchinfo import summary

class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    def __init__(self,
                 img_size:int=224, 
                 in_channels:int=3, 
                 patch_size:int=16, 
                 num_transformer_layers:int=12,
                 embedding_dim:int=768, 
                 mlp_size:int=3072, 
                 num_heads:int=12, 
                 attn_dropout:float=0, 
                 mlp_dropout:float=0.1, 
                 embedding_dropout:float=0.1, 
                 num_classes:int=1000): 
        super().__init__() 

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=768,
                                                                            nhead=num_heads,
                                                                            dim_feedforward=mlp_size,
                                                                            dropout=mlp_dropout,
                                                                            activation='gelu',
                                                                            batch_first=True,
                                                                            norm_first=True) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
    
class PatchEmbedding(nn.Module):
    """Create 1D Sequence lernable embedding vector from a 2D input image

    Args:
        in_channels (int): Nunber of Color Channels. Defaults to 3
        patch_size (int): Target size for each patch. Defaults to 8
        embedding_dim (int): Size of image embedding. Defaults to 768 (ViT base) 
    """

    def __init__(self,
                 in_channels:int = 3,
                 patch_size:int = 8,
                 embedding_dim:int = 768
                 ):
        
        super().__init__()
        
        self.patch_size = patch_size 

        # Layer to create patch embeddings
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # Layer to flatten the flatten the feature map dim. to a single vector
        self.flatten = nn.Flatten(
            start_dim=2, end_dim=3
        )
    
    def forward(self, x):
        image_size = x.shape[-1]
        assert image_size % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_size}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)
    
if __name__ == "__main__":

    # Creating a test batch of images
    test_batch = torch.randn((1, 3, 224, 224))
    in_channels = test_batch.shape[1]
    height, width = test_batch.shape[2], test_batch.shape[3]
    embedding_dimension = 768
    patch_size = 16

    # Testing the PatchEmbedding Module
    test_patcher = PatchEmbedding(in_channels=in_channels,
                                  patch_size=patch_size,
                                   embedding_dim=embedding_dimension
                                )
    
    # Apply the PatchEmbedding module
    output_patch_embedding = test_patcher(test_batch)

    # Assertion checks
    assert test_batch.shape == torch.Size([1, 3, 224, 224]), "\033[1;31;40m Input image shape does not match"
    assert output_patch_embedding.shape == torch.Size([1, 196, 768]), "\033[1;31;40m Output patch embedding shape does not match"
    # print("\033[1;32m PatchEmbedding: Tests passed!\n")
    

    # # Testing ViT
    vit = ViT(num_classes=10)
    try:
        result = vit(test_batch)
        print("\033[1;32m ViT: Tests passed!")
    except:
        print("\033[1;31;40m ViT Failed")

    summary(model=vit,
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )