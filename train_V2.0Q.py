import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ================= [M24 CONFIGURATION v5.2] ================= #
DTYPE = torch.complex64
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(">> ACCELERATOR: Apple MPS (Metal Performance Shaders) Detected.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(">> ACCELERATOR: NVIDIA CUDA Detected.")
else:
    DEVICE = torch.device("cpu")
    print(">> ACCELERATOR: CPU (Fallback).")

# [UPGRADE B] 维度扩张
Q_DIM = 256       # 提升至 256 (复数维度)
BATCH_SIZE = 64   # 增大 Batch
SEQ_LEN = 32
LEARNING_RATE = 1e-3
STEPS = 300       # 增加步数以确保收敛

# ================= [DATA INJECTION: SHAKESPEARE] ================= #
# [UPGRADE A] 真实文明数据
RAW_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
"""

# 简单的字符级 Tokenizer
chars = sorted(list(set(RAW_TEXT)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data_indices = torch.tensor([char_to_idx[c] for c in RAW_TEXT], dtype=torch.long)

print(f"📚 [DATA] Corpus Loaded. Length: {len(RAW_TEXT)} chars | Vocab Size: {vocab_size}")
print(f"   Sample: {RAW_TEXT[:42]}...")

def get_batch():
    """随机采样一段文本"""
    ix = torch.randint(len(data_indices) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data_indices[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data_indices[i+1:i+SEQ_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ================= [M24 UTILITY KERNELS] ================= #

def complex_normalize(x, dim=-1, eps=1e-9):
    real_sq = x.real.pow(2)
    imag_sq = x.imag.pow(2)
    mag_sq = torch.sum(real_sq + imag_sq, dim=dim, keepdim=True)
    denom = torch.sqrt(mag_sq + eps)
    return x / denom

# ================= [AXIOMATIC QUANTUM MODULES] ================= #

class UnitaryLinear(nn.Module):
    """Cayley Transform Implementation"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim, dim, dtype=DTYPE) * 0.02) # 小初始化
        self.register_buffer("identity", torch.eye(dim, dtype=DTYPE))

    def forward(self, x):
        A = self.weight - self.weight.conj().transpose(-2, -1)
        A_cpu = A.cpu()
        I_cpu = self.identity.cpu()
        numerator = I_cpu + A_cpu
        denominator = I_cpu - A_cpu
        U_cpu = torch.linalg.inv(denominator) @ numerator
        U = U_cpu.to(x.device)
        return x @ U.transpose(-1, -2)

class QuantumStatePreparation(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.phase = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        mag = self.emb(x)
        phase = self.phase(x)
        complex_state = mag.type(DTYPE) * torch.exp(1j * phase.type(DTYPE))
        return complex_normalize(complex_state, dim=-1)

class EntanglementLayer(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dilation = dilation
        self.mixer = UnitaryLinear(dim)

    def forward(self, x):
        pad = self.dilation
        past_x = torch.roll(x, shifts=pad, dims=1)
        past_x[:, :pad, :] = 0
        superposition = (x + past_x)
        superposition = complex_normalize(superposition, dim=-1)
        return self.mixer(superposition)

class QuantumMeasurement(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.measure_vectors = nn.Parameter(torch.randn(vocab_size, dim, dtype=DTYPE))

    def forward(self, x):
        M_norm = complex_normalize(self.measure_vectors, dim=-1)
        overlap = torch.matmul(x, M_norm.conj().T)
        logits = (torch.abs(overlap) ** 2)
        # 温度系数 0.05 (更锐利，加速收敛显示)
        probs = F.softmax(logits / 0.05, dim=-1)
        return probs

class H2Q_Quantum_Universe(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = QuantumStatePreparation(vocab_size, Q_DIM)
        # 增加深度以应对更复杂的语言任务
        self.layers = nn.ModuleList([
            EntanglementLayer(Q_DIM, dilation=d) for d in [1, 2, 4, 8]
        ])
        self.measurement = QuantumMeasurement(Q_DIM, vocab_size)

    def forward(self, x):
        psi = self.prep(x)
        for layer in self.layers:
            psi_evolved = layer(psi)
            psi = complex_normalize(psi + psi_evolved, dim=-1)
        probs = self.measurement(psi)
        return torch.log(probs + 1e-9), probs # 返回 probs 用于可视化

# ================= [EXECUTION & VISUALIZATION] ================= #

def calculate_von_neumann_entropy(state_vector):
    with torch.no_grad():
        _, s, _ = torch.linalg.svd(state_vector.cpu())
        s = s**2
        s = s / (torch.sum(s, dim=-1, keepdim=True) + 1e-9)
        entropy = -torch.sum(s * torch.log(s + 1e-9), dim=-1)
    return entropy.mean().item()

def visualize_results(loss_history, entropy_history, wave_snapshots, target_char_idx):
    """
    [UPGRADE C] 综合可视化
    """
    plt.figure(figsize=(18, 6))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='NLL Loss', color='cyan', linewidth=2)
    plt.title('Quantum State Collapse (Learning)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 2. Entropy
    plt.subplot(1, 3, 2)
    plt.plot(entropy_history, label='Entanglement Entropy', color='magenta', linewidth=2)
    plt.title('System Complexity (Entropy)')
    plt.xlabel('Step')
    plt.grid(True, alpha=0.3)
    
    # 3. Wavefunction Heatmap
    plt.subplot(1, 3, 3)
    # wave_snapshots: [Steps, Vocab_Size]
    snapshots = np.array(wave_snapshots)
    sns.heatmap(snapshots.T, cmap='inferno', cbar=True)
    plt.title(f'Wavefunction Probability Density\nTarget: "{idx_to_char[target_char_idx]}" (Index {target_char_idx})')
    plt.xlabel('Training Snapshot (x10 steps)')
    plt.ylabel('Vocabulary Index')
    
    plt.tight_layout()
    plt.savefig('quantum_civilization_viz.png')
    print("\n📊 [VISUALIZATION] Plot saved to 'quantum_civilization_viz.png'")
    plt.show()

def main():
    print("🌌 [M24-OMEGA-X+] Civilization Protocol Initiated")
    print(f"   Target: Shakespeare (Hamlet) | Device: {DEVICE}")
    print(f"   Quantum Dim: {Q_DIM} (Complex) | Vocab: {vocab_size}")
    
    model = H2Q_Quantum_Universe().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 60)
    
    loss_history = []
    entropy_history = []
    wave_snapshots = [] # 存储特定时刻的概率分布
    
    # 选定一个固定的测试序列来观察波函数演化
    # 例如: "To be, or not to b" -> 预测 "e"
    test_seq_str = "To be, or not to b"
    test_seq = torch.tensor([[char_to_idx[c] for c in test_seq_str]], dtype=torch.long).to(DEVICE)
    target_char = 'e'
    target_idx = char_to_idx[target_char]
    
    start_time = time.time()
    
    for step in range(STEPS):
        optimizer.zero_grad()
        
        # 获取真实数据 Batch
        x_batch, y_batch = get_batch()
        
        log_probs, _ = model(x_batch)
        loss = F.nll_loss(log_probs.view(-1, vocab_size), y_batch.view(-1))
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # 监控与采样
        if step % 10 == 0:
            # 1. 计算熵
            current_psi = model.prep(x_batch)
            entropy = calculate_von_neumann_entropy(current_psi)
            entropy_history.append(entropy)
            
            # 2. [UPGRADE C] 捕获波函数快照
            with torch.no_grad():
                _, probs = model(test_seq)
                # 取最后一个时间步的概率分布 [1, Seq, Vocab] -> [Vocab]
                last_token_probs = probs[0, -1, :].cpu().numpy()
                wave_snapshots.append(last_token_probs)
            
            elapsed = time.time() - start_time
            print(f"Step {step:03d} | Loss: {loss_val:.4f} | Entropy: {entropy:.4f} | Time: {elapsed:.1f}s")
        else:
            entropy_history.append(entropy_history[-1] if entropy_history else 0)

    print("-" * 60)
    
    # 最终生成测试
    print("📝 [GENERATION TEST]")
    prompt = "To be, or not to "
    input_ids = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _, probs = model(input_ids)
            # 贪婪采样
            next_token = torch.argmax(probs[0, -1, :]).item()
            print(idx_to_char[next_token], end="", flush=True)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
    print("\n" + "-" * 60)
    
    visualize_results(loss_history, entropy_history, wave_snapshots, target_idx)

if __name__ == "__main__":
    main()
