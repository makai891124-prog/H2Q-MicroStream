import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
import os
import sys
import requests

# ==========================================
# 0. H2Q-RollingHorizon: ICU 监控版
# ==========================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    device_compute = torch.device("cuda")
    device_structure = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🌊 H2Q-ICU Monitor Online: {gpu_name}")
    print(f"   [Mode: Deep Analysis] [Metrics: Grad/VRAM/TPS/Diff]")
else:
    device_compute = torch.device("cpu")
    device_structure = torch.device("cpu")

CONFIG = {
    'dim': 768,           
    'factor_size': 32,
    'fixed_rank': 8,       
    'depth': 12,           
    'seq_len': 128,        
    'batch_size': 24,      
    'grad_accum_steps': 1, 
    'lr': 6e-4,            
    'weight_decay': 0.02,
    'dropout_rate': 0.1,
    'axiom_lambda': 0.1,
    
    'total_chunks': 10000, 
    'chunk_size_mb': 10,   
    
    'checkpoint_path': 'h2q_rolling.pt',
    'best_model_path': 'h2q_rolling_best.pt',
    'data_dir': 'data_tinystories',
}

# ==========================================
# 1. 仪表盘工具 (Dashboard Utils)
# ==========================================
def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}/{reserved:.2f}GB"
    return "N/A"

def color_loss(val, train):
    # 如果验证集 Loss 比 训练集还低，绿色（泛化极好）
    # 如果验证集 Loss 高出 0.5 以上，红色（过拟合风险）
    diff = val - train
    if diff < 0: return f"\033[92m{val:.4f}\033[0m" # Green
    if diff > 0.5: return f"\033[91m{val:.4f}\033[0m" # Red
    return f"{val:.4f}" # White

# ==========================================
# 2. 轮动加载器
# ==========================================
class RollingWheelLoader:
    def __init__(self, config, resume_offset=0):
        self.chunk_size = config['chunk_size_mb'] * 1024 * 1024
        self.batch_size = config['batch_size']
        self.data_dir = config['data_dir']
        self.file_path = os.path.join(self.data_dir, 'TinyStories-train.txt')
        self._prepare_data()
        
        if not os.path.exists(self.file_path):
            print(f"❌ 错误：找不到数据文件 {self.file_path}")
            sys.exit(1)

        self.file = open(self.file_path, 'r', encoding='utf-8', errors='ignore')
        self.current_offset = resume_offset
        if resume_offset > 0:
            print(f"🔖 [时间之轮] 回溯至偏移量: {resume_offset / 1024 / 1024:.2f} MB")
            self.file.seek(resume_offset)
            
    def _prepare_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        if os.path.exists(self.file_path): return
        print(f"📦 正在下载数据集...")
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
        try:
            r = requests.get(url, stream=True)
            with open(self.file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): 
                    if chunk: f.write(chunk)
            print("✅ 下载完成！")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)

    def load_next_chunk_tensor(self):
        try:
            text = self.file.read(self.chunk_size)
        except Exception: return None
        
        if not text:
            print("🔄 [轮回] 世界尽头，重返原点...")
            self.file.seek(0)
            self.current_offset = 0
            text = self.file.read(self.chunk_size)
            
        self.current_offset += len(text)
        data_list = [ord(c) if ord(c) < 256 else 32 for c in text]
        data = torch.tensor(data_list, dtype=torch.long)
        
        num_batches = len(data) // self.batch_size
        valid_len = num_batches * self.batch_size
        if valid_len == 0: return None 
        
        return data[:valid_len].view(self.batch_size, num_batches).contiguous().to(device_compute)

    def get_bookmark(self): return self.file.tell()
    def decode(self, l): return ''.join([chr(i) if i > 0 else '' for i in l])

# ==========================================
# 3. H2Q 核心组件
# ==========================================
class WaveStructureBank(nn.Module):
    def __init__(self, num_blocks, rank):
        super().__init__()
        self.sub_blocks = num_blocks // 4; self.rank = rank
        self.factors_A = nn.Parameter(torch.zeros(rank, 4, self.sub_blocks, self.sub_blocks, device=device_structure))
        with torch.no_grad():
            for r in range(rank):
                c = torch.randn(4, self.sub_blocks, self.sub_blocks, device=device_structure)
                for i in range(4): nn.init.orthogonal_(c[i])
                self.factors_A.data[r] = c * ((r+1)**-0.5)
    def get_factors(self): return self.factors_A

class BalancedHamiltonLayer(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.dim, self.factor_size, self.structure_bank = dim, factor_size, structure_bank
        self.factors_B = nn.Parameter(torch.zeros(rank, factor_size, factor_size, device=device_compute))
        self.bias = nn.Parameter(torch.zeros(dim, device=device_compute))
        with torch.no_grad():
            for r in range(rank):
                b = torch.randn(factor_size, factor_size, device=device_compute)
                nn.init.orthogonal_(b)
                self.factors_B.data[r] = b * ((r+1)**-0.5)
    def _construct_hamilton(self, A):
        r,i,j,k = A[:,0], A[:,1], A[:,2], A[:,3]
        return torch.cat([torch.cat([r,-i,-j,-k],2), torch.cat([i,r,-k,j],2), torch.cat([j,k,r,-i],2), torch.cat([k,-j,i,r],2)],1)
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, 4*self.structure_bank.sub_blocks, self.factor_size)
        A = self.structure_bank.get_factors().to(dtype=x.dtype)
        wav = torch.einsum('nsi,rji->rnsj', x_flat, self.factors_B.to(dtype=x.dtype))
        out = torch.einsum('rnsj,rks->nkj', wav, self._construct_hamilton(A))
        return out.reshape(B,T,D) + self.bias
    def ortho_loss(self):
        loss = 0
        for p in self.factors_B: loss += torch.norm(torch.mm(p.float().t(), p.float()) - torch.eye(p.shape[1], device=device_compute))
        return loss

class QuaternionAttention(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank, num_heads=8):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = dim // num_heads; self.scale = self.head_dim**-0.5
        self.q_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.k_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.v_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.o_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
    def forward(self, x):
        B,T,C = x.shape
        # 修复维度的正确 Attention 计算 [B, H, T, T]
        q = self.q_proj(x).view(B,T,self.num_heads,-1).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.num_heads,-1).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.num_heads,-1).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * self.scale
        mask = torch.triu(torch.ones(T,T,device=device_compute)*float('-inf'),1)
        att = F.softmax(att+mask, dim=-1)
        y = (att @ v).transpose(1,2).reshape(B,T,C)
        return self.o_proj(y)
    def ortho_loss(self): return self.q_proj.ortho_loss()+self.k_proj.ortho_loss()+self.v_proj.ortho_loss()+self.o_proj.ortho_loss()

class HolographicReversibleBlock(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.half = dim // 2
        self.n1 = nn.LayerNorm(self.half); self.attn = QuaternionAttention(self.half, factor_size, structure_bank, rank)
        self.n2 = nn.LayerNorm(self.half); self.f1 = BalancedHamiltonLayer(self.half, factor_size, structure_bank, rank)
        self.act = nn.GELU(); self.f2 = BalancedHamiltonLayer(self.half, factor_size, structure_bank, rank)
    def f(self, x): return self.attn(self.n1(x))
    def g(self, x): return self.f2(self.act(self.f1(self.n2(x))))
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + checkpoint.checkpoint(self.f, x2, use_reentrant=False)
        y2 = x2 + checkpoint.checkpoint(self.g, y1, use_reentrant=False)
        return torch.cat([y1,y2], dim=-1)
    def ortho_loss(self): return self.attn.ortho_loss() + self.f1.ortho_loss() + self.f2.ortho_loss()

class H2Q_Transformer(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.bank = WaveStructureBank(config['dim']//2//config['factor_size'], config['fixed_rank'])
        self.emb = nn.Embedding(vocab_size, config['dim'])
        self.pos = nn.Parameter(torch.randn(1, config['seq_len'], config['dim'])*0.02)
        self.drop = nn.Dropout(config['dropout_rate'])
        self.layers = nn.ModuleList([HolographicReversibleBlock(config['dim'], config['factor_size'], self.bank, config['fixed_rank']) for _ in range(config['depth'])])
        self.head = nn.Linear(config['dim'], vocab_size)
    def forward(self, x, targets=None):
        x = self.drop(self.emb(x) + self.pos[:, :x.size(1), :])
        ol = torch.tensor(0.0, device=device_compute)
        for l in self.layers: 
            x = l(x)
            ol = ol + l.ortho_loss()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(self.head(x).reshape(-1, 256), targets.reshape(-1)) + self.config['axiom_lambda']*ol*0.01
        return self.head(x), loss, x.norm(dim=-1).mean()
    @torch.no_grad()
    def generate(self, idx, new):
        for _ in range(new):
            idx_cond = idx[:, -self.config['seq_len']:]
            logits, _, _ = self(idx_cond)
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits[:,-1,:], dim=-1), 1)), dim=1)
        return idx

# ==========================================
# 3. 深度监控训练循环
# ==========================================
def train_rolling_system():
    resume_offset = 0
    chunk_counter = 0
    best_loss = float('inf')
    
    model = H2Q_Transformer(256, CONFIG).to(device_compute)
    opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')

    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"🔄 恢复存档: {CONFIG['checkpoint_path']}")
        ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device_compute, weights_only=False)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        resume_offset = ckpt.get('dataset_offset', 0)
        chunk_counter = ckpt.get('chunk_counter', 0)
        best_loss = ckpt.get('best_loss', float('inf'))

    loader = RollingWheelLoader(CONFIG, resume_offset)
    if not os.path.exists(loader.file_path): return

    print("⏳ [Init] 加载初始时间块 (Chunk T)...")
    current_chunk_data = loader.load_next_chunk_tensor()
    if current_chunk_data is None: return
    
    print("🚀 启动深度监控 (Deep Monitor Active)...")
    
    try:
        model.train()
        while chunk_counter < CONFIG['total_chunks']:
            t0 = time.time()
            print(f"\n" + "="*50)
            print(f"🧩 CHUNK {chunk_counter}: 开始加载未来数据...")
            
            future_chunk_data = loader.load_next_chunk_tensor()
            if future_chunk_data is None: break 
            
            # --- 验证阶段 ---
            print(f"   🔮 验证未来 (Validation)...")
            model.eval()
            val_loss_accum = 0; val_steps = 0
            eval_limit = future_chunk_data.size(1) // 5 
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                for i in range(0, eval_limit, CONFIG['seq_len']):
                    if i + CONFIG['seq_len'] + 1 > future_chunk_data.size(1): break
                    vx = future_chunk_data[:, i : i+CONFIG['seq_len']]
                    vy = future_chunk_data[:, i+1 : i+CONFIG['seq_len']+1]
                    _, vl, _ = model(vx, vy)
                    val_loss_accum += vl.item(); val_steps += 1
            avg_val_loss = val_loss_accum / (val_steps + 1e-6)
            model.train()
            print(f"   📊 验证结果: Val Loss = {avg_val_loss:.4f}")
            
            # --- 训练阶段 ---
            print(f"   🔥 训练当下 (Training)...")
            train_loss_accum = 0; train_steps = 0; energy_val = 0
            total_train_steps = current_chunk_data.size(1) // CONFIG['seq_len']
            chunk_start_time = time.time()
            
            for i in range(0, current_chunk_data.size(1), CONFIG['seq_len']):
                step_start = time.time()
                if i + CONFIG['seq_len'] + 1 > current_chunk_data.size(1): break
                
                x = current_chunk_data[:, i : i+CONFIG['seq_len']]
                y = current_chunk_data[:, i+1 : i+CONFIG['seq_len']+1]
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _, loss, energy = model(x, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                # 捕获梯度范数
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                
                train_loss_accum += loss.item(); energy_val = energy.item(); train_steps += 1
                
                # 💓 实时心跳包 (每50步刷新一次)
                if train_steps % 50 == 0:
                    step_time = (time.time() - step_start) * 1000 # ms
                    tps = (CONFIG['batch_size'] * CONFIG['seq_len']) / (time.time() - step_start)
                    vram = get_vram_usage()
                    # 动态清除行显示
                    sys.stdout.write(f"\r      ⚡ Step {train_steps:4d}/{total_train_steps} | Loss: {loss.item():.4f} | Grad: {total_norm:.2f} | Energy: {energy_val:.1f} | Speed: {step_time:.0f}ms ({int(tps)} tok/s) | VRAM: {vram}")
                    sys.stdout.flush()
                
            avg_train_loss = train_loss_accum / (train_steps + 1e-6)
            current_chunk_data = future_chunk_data
            chunk_counter += 1
            
            total_time = time.time() - t0
            colored_val = color_loss(avg_val_loss, avg_train_loss)
            
            # 🏁 Chunk 总结
            print(f"\n   ✅ Chunk {chunk_counter} 完成 Summary:")
            print(f"      Train: {avg_train_loss:.4f} | Val: {colored_val} | Diff: {avg_val_loss-avg_train_loss:+.4f}")
            print(f"      Time: {total_time:.1f}s | Progress: {loader.get_bookmark()/1024/1024:.1f} MB")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({'model': model.state_dict(), 'config': CONFIG}, CONFIG['best_model_path'])
            
            # 每个 Chunk 都存，安全第一
            ckpt = {'chunk_counter': chunk_counter, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'best_loss': best_loss, 'dataset_offset': loader.get_bookmark()}
            torch.save(ckpt, CONFIG['checkpoint_path'])
            
            # 每 5 个 Chunk 看一次梦境
            if chunk_counter % 5 == 0:
                print("\n📜 [Thought Stream]:")
                model.eval()
                with torch.no_grad():
                    ctx = torch.tensor([[ord('T')]], dtype=torch.long, device=device_compute)
                    out = model.generate(ctx, 200)
                    print(loader.decode(out[0].tolist()))
                model.train()

    except KeyboardInterrupt:
        print("\n🛑 监测到中断，紧急保存...")
        ckpt = {'chunk_counter': chunk_counter, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'best_loss': best_loss, 'dataset_offset': loader.get_bookmark()}
        torch.save(ckpt, CONFIG['checkpoint_path'])

if __name__ == "__main__":
    train_rolling_system()