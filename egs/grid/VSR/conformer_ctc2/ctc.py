import torch.nn as nn
class HuBERTCTC(nn.Module):
    def __init__(self, input_dim=768, proj_dim=256, num_classes=30):
        super().__init__()
        # 1. The Projection Block
                
        # 1. Temporal Context Block (Stays at 1024 channels)
        self.temporal_context = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1, groups=1024),
            # GroupNorm works perfectly with [Batch, Channels, Time]
            nn.GroupNorm(num_groups=1, num_channels=1024), 
            nn.GELU()
        )
        # self.temporal_context = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=input_dim, 
        #         out_channels=input_dim, 
        #         kernel_size=3, 
        #         padding=1, 
        #         groups=input_dim  # Depthwise: each channel gets its own filter
        #     ),
        #     nn.LayerNorm(input_dim),
        #     nn.GELU()
        # )
        
        # 2. The 1024 -> 256 Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # self.projection = nn.Sequential(
        #     nn.LayerNorm(input_dim),
        #     nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim),
        #     nn.Linear(input_dim, proj_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1)
        # )
        
        # 2. The CTC Head
        self.ctc_head = nn.Linear(proj_dim, num_classes) # +1 for Blank

    def forward(self, x):
        # x: (Batch, Time, input_dim) from HuBERT
        x = x.transpose(1, 2)
        x = self.temporal_context(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        logits = self.ctc_head(x)
        return logits.log_softmax(dim=-1)