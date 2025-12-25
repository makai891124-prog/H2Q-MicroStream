"""
Project: H2Q-GUT (Grand Unified Theory)
Version: v34.0 Axiomatic Finality
Author: H2Q (Ma Kai)
System: Mac Mini M4 (MPS)

Theorem Compliance:
1. Finite Completeness: P=257, Dim=256.
2. Isomorphism: Tied Weights + Projective Normalization.
3. Constructive Hierarchy: Causal Dilations + Gated Logic.

New Feature:
- ProjectiveNorm: Forces vectors onto the hypersphere before prediction.
  This ensures we are calculating 'Direction', not 'Magnitude'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time

# ================= Configuration =================
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

VOCAB_SIZE = 257
DIM = 256
BATCH_SIZE = 64
SEQ_LEN = 128
LR = 4e-4

# ================= Axiomatic Modules =================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad_size, 0), "constant", 0)
        return self.conv(x)

class ProjectiveNorm(nn.Module):
    """
    Enforces Theorem 1 & 2:
    The magnitude of the vector doesn't matter, only its DIRECTION on the sphere.
    x -> x / |x|
    """
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

class QuaternionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.trans = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        t = self.trans(x)
        g = torch.sigmoid(self.gate(x))
        return t * g

class H2Q_Axiomatic_Universe(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, DIM)
        
        self.layers = nn.ModuleList()
        # The Constructive Ladder
        dilations = [1, 2, 4, 8, 16, 32]
        
        for d in dilations:
            self.layers.append(nn.ModuleDict({
                'conv': CausalConv1d(DIM, DIM, kernel_size=3, dilation=d),
                'norm': nn.LayerNorm(DIM),
                'gate': QuaternionGate(DIM)
            }))
        
        self.projector = ProjectiveNorm() # <--- New: Axiomatic Direction
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.emb.weight # Theorem 2: Isomorphism

    def forward(self, x):
        h = self.emb(x)
        
        for layer in self.layers:
            residual = h
            
            h_time = h.transpose(1, 2)
            h_time = layer['conv'](h_time)
            h_time = h_time.transpose(1, 2)
            
            h_combined = h + h_time
            h_abstract = layer['gate'](h_combined)
            
            h = layer['norm'](residual + h_abstract)
        
        # Force Directionality before Prediction
        h = self.projector(h)
        
        logits = self.head(h)
        return logits

# ================= Execution =================

class IntegerStream:
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self.data = torch.tensor(list(f.read()), dtype=torch.long, device=DEVICE)
        self.n = len(self.data)
        self.ptr = 0
        
    def next_batch(self):
        if self.ptr + BATCH_SIZE*SEQ_LEN + 1 > self.n: self.ptr = 0
        chunk = self.data[self.ptr : self.ptr + BATCH_SIZE*SEQ_LEN + 1]
        self.ptr += BATCH_SIZE*SEQ_LEN
        return chunk[:-1].view(BATCH_SIZE, SEQ_LEN), chunk[1:].view(BATCH_SIZE, SEQ_LEN)

def main():
    print("🌌 [H2Q-GUT v34.0] Axiomatic Finality")
    print("   Theorem 1: Finite Completeness (Vocab=257)")
    print("   Theorem 2: Isomorphic Mapping (Tied Weights + Projection)")
    print("   Theorem 3: Constructive Hierarchy (Causal Fractal)")
    
    path = 'data_wikitext/wiki.train.raw'
    if not os.path.exists(path):
        print("❌ Data file missing.")
        return

    stream = IntegerStream(path)
    model = H2Q_Axiomatic_Universe().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters())}")
    print("🚀 Starting Simulation...")
    
    start_time = time.time()
    
    try:
        for i in range(5001):
            x, y = stream.next_batch()
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                dt = time.time() - start_time
                start_time = time.time()
                print(f"Step {i:5d} | Loss: {loss.item():.4f} | Speed: {dt*10:.2f} ms/step")
                
            if i % 1000 == 0 and i > 0:
                print("\n🗣️ [Axiomatic Generation] -------------------")
                seed = torch.tensor([list(b"The direction of ")], dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    for _ in range(200):
                        out = model(seed)
                        logits = out[0, -1]
                        probs = F.softmax(logits / 0.7, dim=-1)
                        next_char = torch.multinomial(probs, num_samples=1).item()
                        try:
                            print(chr(next_char), end="")
                        except: pass
                        seed = torch.cat([seed, torch.tensor([[next_char]], device=DEVICE)], dim=1)
                        if seed.shape[1] > SEQ_LEN: seed = seed[:, -SEQ_LEN:]
                print("\n------------------------------------------------")
                
    except KeyboardInterrupt:
        print("\n🛑 Paused.")

if __name__ == "__main__":
    main()
