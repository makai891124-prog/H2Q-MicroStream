import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
import os
import sys
import shutil
import glob
import pandas as pd
import threading
import queue
import platform
import traceback
import random

# DeepSeek 监督器 (可选，密钥缺失时自动禁用)
try:
    from deepseek_supervisor import DeepSeekSupervisor
    _SUPERVISOR_AVAILABLE = True
except ImportError:
    _SUPERVISOR_AVAILABLE = False

# ==========================================
# 0. H2Q-RollingHorizon: TF32 High-Precision Edition
# ==========================================
# 🔥 核心精度设置：启用 TensorFloat-32 (TF32)
# 这在 Ampere/Ada 架构(4070Ti)上提供了接近 FP32 的精度和接近 FP16 的速度
torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    device_compute = torch.device("cuda")
    device_structure = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🌊 H2Q-HPC Monitor Online: {gpu_name}")
    print(f"   [Mode: TF32 Physical Precision] [No AMP] [Dual-Stream]")
else:
    device_compute = torch.device("cpu")
    device_structure = torch.device("cpu")

# ==========================================
# ⚙️ 实验配置 (Configuration)
# ==========================================
CONFIG = {
    # --- 模型架构 (保持不变) ---
    'dim': 768,           
    'factor_size': 32,
    'fixed_rank': 8,       
    'depth': 12,           
    'seq_len': 128,        
    'batch_size': 24,      
    'dropout_rate': 0.1,
    'axiom_lambda': 0.1,
    
    # --- 优化器 ---
    'grad_accum_steps': 1, 
    'lr': 3e-4,            # 保持低学习率进行精细雕刻
    'weight_decay': 0.02,
    
    # --- 训练流 ---
    'total_chunks': 100000, 
    'chunk_size_mb': 10,    # 保持 10MB 以获得极速反馈
    
    # --- 路径配置 ---
    'checkpoint_path': 'h2q_fineweb.pt',
    'best_model_path': 'h2q_fineweb_best.pt',
    'source_dir': r'E:\Datasets\FineWeb-Edu_Full', 
    'buffer_dir': r'D:\H2Q_Cache_Zone',             

    # --- DeepSeek 监督参数 ---
    # 每隔多少个 chunk 触发一次 DeepSeek 监督注入 (0 = 禁用)
    'deepseek_supervise_every': 10,
    # 模型生成采样长度 (tokens)
    'deepseek_gen_tokens': 256,
}


def _env_int(name, default):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name, default):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def apply_env_overrides(config):
    config['total_chunks'] = _env_int("H2Q_TOTAL_CHUNKS", config['total_chunks'])
    config['chunk_size_mb'] = _env_int("H2Q_CHUNK_SIZE_MB", config['chunk_size_mb'])
    config['batch_size'] = _env_int("H2Q_BATCH_SIZE", config['batch_size'])
    config['seq_len'] = _env_int("H2Q_SEQ_LEN", config['seq_len'])
    config['lr'] = _env_float("H2Q_LR", config['lr'])

    source_dir = os.getenv("H2Q_SOURCE_DIR")
    if source_dir:
        config['source_dir'] = source_dir

    buffer_dir = os.getenv("H2Q_BUFFER_DIR")
    if buffer_dir:
        config['buffer_dir'] = buffer_dir

    checkpoint_path = os.getenv("H2Q_CHECKPOINT_PATH")
    if checkpoint_path:
        config['checkpoint_path'] = checkpoint_path

    best_model_path = os.getenv("H2Q_BEST_MODEL_PATH")
    if best_model_path:
        config['best_model_path'] = best_model_path


apply_env_overrides(CONFIG)

# ==========================================
# 1. 仪表盘工具
# ==========================================
def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}/{reserved:.2f}GB"
    return "N/A"

def color_loss(val, train):
    diff = val - train
    if diff < 0: return f"\033[92m{val:.4f}\033[0m" 
    if diff > 0.5: return f"\033[91m{val:.4f}\033[0m" 
    return f"{val:.4f}" 

# ==========================================
# 2. 双流混合加载器 (AsyncBufferedLoader)
# ==========================================
class AsyncBufferedLoader:
    def __init__(self, config, resume_file_index=0):
        self.chunk_size = config['chunk_size_mb'] * 1024 * 1024
        self.batch_size = config['batch_size']
        self.source_dir = config['source_dir']
        self.buffer_dir = config['buffer_dir']
        
        self.injection_dir = os.path.join(self.buffer_dir, "Injection")
        if not os.path.exists(self.injection_dir): os.makedirs(self.injection_dir)
        
        print(f"🔍 正在扫描主源目录: {self.source_dir} ...")
        self.file_list = sorted(glob.glob(os.path.join(self.source_dir, "**/*.parquet"), recursive=True))
        if not self.file_list:
            print(f"❌ 错误：在 {self.source_dir} 未找到 .parquet 文件")
            sys.exit(1)
        print(f"📚 发现 {len(self.file_list)} 个主数据文件。")
        
        self.current_file_index = int(resume_file_index)
        self.queue = queue.Queue(maxsize=3) 
        self.stop_event = threading.Event()
        self.buffer_integers = []
        
        self.loader_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.loader_thread.start()

    def _clean_buffer_dir(self):
        for f in os.listdir(self.buffer_dir):
            full_path = os.path.join(self.buffer_dir, f)
            if os.path.isfile(full_path): 
                try: os.remove(full_path)
                except: pass

    def _ingest_injection_files(self):
        """优先吞噬注入数据"""
        injection_files = sorted(glob.glob(os.path.join(self.injection_dir, "*.parquet")))
        if not injection_files: return False
            
        print(f"💉 [注入] 发现 {len(injection_files)} 个意识碎片，正在融合...")
        for inj_file in injection_files:
            try:
                df = pd.read_parquet(inj_file, columns=['text'])
                try: os.remove(inj_file) 
                except: pass
                
                texts = df['text'].dropna().astype(str).tolist()
                for text in texts:
                    b = text.encode('utf-8', errors='ignore') + b'\0'
                    self.buffer_integers.extend(b)
            except Exception as e:
                print(f"⚠️ 注入读取失败: {e}")
        return True

    def _process_parquet_content(self, file_path):
        try:
            df = pd.read_parquet(file_path, columns=['text'])
            try: os.remove(file_path)
            except: pass
            
            texts = df['text'].dropna().astype(str).tolist()
            
            for i, text in enumerate(texts):
                if self.stop_event.is_set(): break
                
                if i % 1000 == 0:
                    self._ingest_injection_files()

                b = text.encode('utf-8', errors='ignore') + b'\0'
                self.buffer_integers.extend(b)
                
                if len(self.buffer_integers) >= self.chunk_size:
                    data_tensor = torch.tensor(self.buffer_integers[:self.chunk_size], dtype=torch.long)
                    self.queue.put(data_tensor) 
                    self.buffer_integers = self.buffer_integers[self.chunk_size:]
                    
        except Exception as e:
            print(f"⚠️ 文件处理错误 ({os.path.basename(file_path)}): {e}")

    def _background_worker(self):
        if self.current_file_index == 0: self._clean_buffer_dir()
        
        while not self.stop_event.is_set():
            self._ingest_injection_files()
            
            if self.current_file_index >= len(self.file_list):
                print("🔄 [轮回] 数据集遍历完毕，重置索引...")
                self.current_file_index = 0
            
            src_path = self.file_list[self.current_file_index]
            filename = os.path.basename(src_path)
            buffer_path = os.path.join(self.buffer_dir, filename)
            
            try:
                if not os.path.exists(buffer_path):
                    shutil.copy2(src_path, buffer_path)
                self._process_parquet_content(buffer_path)
                self.current_file_index += 1
            except Exception as e:
                print(f"⚠️ 主循环错误: {e}")
                self.current_file_index += 1
                time.sleep(1)

    def load_next_chunk_tensor(self):
        try:
            data = self.queue.get(timeout=120)
            num_batches = len(data) // self.batch_size
            valid_len = num_batches * self.batch_size
            if valid_len == 0: return self.load_next_chunk_tensor()
            return data[:valid_len].view(self.batch_size, num_batches).contiguous().to(device_compute)
        except queue.Empty:
            print("❌ 数据加载超时")
            return None

    def get_bookmark(self): return self.current_file_index
    def decode(self, l):
        valid_bytes = bytes([i for i in l if i > 0])
        return valid_bytes.decode('utf-8', errors='ignore')

# ==========================================
# 3. H2Q 核心架构 (保持不变)
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
# 4. 深度监控训练循环 (TF32 高精版)
# ==========================================
def sanitize_state_dict(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."): new_dict[k[10:]] = v
        else: new_dict[k] = v
    return new_dict

def train_rolling_system():
    resume_file_index = 0
    chunk_counter = 0
    best_loss = float('inf')
    
    # 1. 初始化模型
    model = H2Q_Transformer(256, CONFIG).to(device_compute)
    
    if platform.system() == "Windows":
        print("⚠️ 检测到 Windows 系统，禁用 torch.compile。")
    else:
        try:
            print("⚡ 正在编译计算图 (Torch Compile)...")
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"⚠️ 编译失败: {e}")

    opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # 🔥 移除 GradScaler (FP32 不需要)
    # scaler = torch.amp.GradScaler('cuda') 

    # 2. 尝试加载存档
    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"🔄 恢复存档: {CONFIG['checkpoint_path']}")
        try:
            ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device_compute, weights_only=False)
            clean_state_dict = sanitize_state_dict(ckpt['model'])
            model.load_state_dict(clean_state_dict)
            opt.load_state_dict(ckpt['optimizer'])
            
            saved_offset = ckpt.get('dataset_offset', 0)
            if saved_offset > 1000000: resume_file_index = 0
            else: resume_file_index = saved_offset
                
            chunk_counter = ckpt.get('chunk_counter', 0)
            best_loss = ckpt.get('best_loss', float('inf'))
            print("✅ 存档加载成功！")
        except Exception as e:
            print(f"❌ 存档加载严重错误: {e}")
            print("   将从头开始训练...")
    
    # 强制应用学习率
    for param_group in opt.param_groups:
        param_group['lr'] = CONFIG['lr']
    print(f"🔧 学习率已强制调整为: {CONFIG['lr']}")

    loader = AsyncBufferedLoader(CONFIG, resume_file_index)

    # 初始化 DeepSeek 监督器
    supervisor = None
    if _SUPERVISOR_AVAILABLE and CONFIG.get('deepseek_supervise_every', 0) > 0:
        injection_dir = os.path.join(CONFIG['buffer_dir'], 'Injection')
        supervisor = DeepSeekSupervisor(
            injection_dir=injection_dir,
            every_n_chunks=CONFIG['deepseek_supervise_every'],
            gen_tokens=CONFIG.get('deepseek_gen_tokens', 256),
        )
    else:
        print("[DeepSeek] ℹ️  监督器未启用。")

    print("⏳ [Init] 等待后台线程准备初始数据 (Chunk T)...")
    current_chunk_data = loader.load_next_chunk_tensor()
    if current_chunk_data is None: 
        print("❌ 无法加载初始数据。")
        input("按回车键退出...") 
        return
    
    print("🚀 启动深度监控 (TF32 High Precision Mode)...")
    
    try:
        model.train()
        while chunk_counter < CONFIG['total_chunks']:
            t0 = time.time()
            print(f"\n" + "="*50)
            print(f"🧩 CHUNK {chunk_counter}: 开始加载未来数据...")
            
            future_chunk_data = loader.load_next_chunk_tensor()
            if future_chunk_data is None: 
                print("🏁 数据集已耗尽。")
                break 
            
            # --- 验证阶段 (移除 autocast) ---
            print(f"   🔮 验证未来 (Validation)...")
            model.eval()
            val_loss_accum = 0; val_steps = 0
            eval_limit = min(future_chunk_data.size(1), 1000 * CONFIG['seq_len'])
            
            with torch.no_grad(): # 🔥 移除 autocast
                for i in range(0, eval_limit, CONFIG['seq_len']):
                    if i + CONFIG['seq_len'] + 1 > future_chunk_data.size(1): break
                    vx = future_chunk_data[:, i : i+CONFIG['seq_len']]
                    vy = future_chunk_data[:, i+1 : i+CONFIG['seq_len']+1]
                    _, vl, _ = model(vx, vy)
                    val_loss_accum += vl.item(); val_steps += 1
            avg_val_loss = val_loss_accum / (val_steps + 1e-6)
            model.train()
            print(f"   📊 验证结果: Val Loss = {avg_val_loss:.4f}")
            
            # --- 训练阶段 (移除 autocast 和 scaler) ---
            print(f"   🔥 训练当下 (Training)...")
            train_loss_accum = 0; train_steps = 0; energy_val = 0
            total_train_steps = current_chunk_data.size(1) // CONFIG['seq_len']
            chunk_start_time = time.time()
            
            for i in range(0, current_chunk_data.size(1), CONFIG['seq_len']):
                step_start = time.time()
                if i + CONFIG['seq_len'] + 1 > current_chunk_data.size(1): break
                
                x = current_chunk_data[:, i : i+CONFIG['seq_len']]
                y = current_chunk_data[:, i+1 : i+CONFIG['seq_len']+1]
                
                # 🔥 直接前向传播 (默认 TF32)
                _, loss, energy = model(x, y)
                
                # 🔥 直接反向传播 (无 scaler)
                loss.backward()
                
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 🔥 直接更新
                opt.step()
                opt.zero_grad()
                
                train_loss_accum += loss.item(); energy_val = energy.item(); train_steps += 1
                
                if train_steps % 50 == 0:
                    step_time = (time.time() - step_start) * 1000 
                    tps = (CONFIG['batch_size'] * CONFIG['seq_len']) / (time.time() - step_start)
                    vram = get_vram_usage()
                    sys.stdout.write(f"\r      ⚡ Step {train_steps:4d}/{total_train_steps} | Loss: {loss.item():.4f} | Grad: {total_norm:.2f} | Energy: {energy_val:.1f} | Speed: {step_time:.0f}ms ({int(tps)} tok/s) | VRAM: {vram}")
                    sys.stdout.flush()
                
            avg_train_loss = train_loss_accum / (train_steps + 1e-6)
            current_chunk_data = future_chunk_data
            chunk_counter += 1
            
            total_time = time.time() - t0
            colored_val = color_loss(avg_val_loss, avg_train_loss)
            
            print(f"\n   ✅ Chunk {chunk_counter} 完成 Summary:")
            print(f"      Train: {avg_train_loss:.4f} | Val: {colored_val} | Diff: {avg_val_loss-avg_train_loss:+.4f}")
            print(f"      File Index: {loader.get_bookmark()} | Time: {total_time:.1f}s")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({'model': model.state_dict(), 'config': CONFIG}, CONFIG['best_model_path'])
            
            ckpt = {
                'chunk_counter': chunk_counter, 
                'model': model.state_dict(), 
                'optimizer': opt.state_dict(), 
                'best_loss': best_loss, 
                'dataset_offset': loader.get_bookmark()
            }
            torch.save(ckpt, CONFIG['checkpoint_path'])
            
            # DeepSeek 监督注入 (在 Thought Stream 之前触发)
            if supervisor is not None:
                supervisor.maybe_supervise(chunk_counter, model, loader, device_compute)

            if chunk_counter % 5 == 0:
                print("\n📜 [Thought Stream - 自由联想]:")
                model.eval()
                with torch.no_grad():
                    seeds = ["I", "The", "Why", "If", "It is", "\n"]
                    seed_str = random.choice(seeds)
                    print(f"✨ 诱导词: [{seed_str}]")
                    ctx = torch.tensor([list(seed_str.encode('utf-8'))], dtype=torch.long, device=device_compute)
                    out = model.generate(ctx, 500)
                    print(loader.decode(out[0].tolist()))
                    print("-" * 50)
                model.train()

    except KeyboardInterrupt:
        print("\n🛑 监测到中断，紧急保存...")
        ckpt = {
            'chunk_counter': chunk_counter, 
            'model': model.state_dict(), 
            'optimizer': opt.state_dict(), 
            'best_loss': best_loss, 
            'dataset_offset': loader.get_bookmark()
        }
        torch.save(ckpt, CONFIG['checkpoint_path'])
        
    except Exception as e:
        print(f"\n❌ 发生严重错误 (CRITICAL ERROR):")
        traceback.print_exc()
        print(f"错误信息: {e}")
        print("💾 尝试紧急保存当前状态...")
        try:
            ckpt = {
                'chunk_counter': chunk_counter, 
                'model': model.state_dict(), 
                'optimizer': opt.state_dict(), 
                'best_loss': best_loss, 
                'dataset_offset': loader.get_bookmark()
            }
            torch.save(ckpt, CONFIG['checkpoint_path'])
            print("✅ 紧急保存成功。")
        except:
            print("❌ 紧急保存失败。")
            
    finally:
        no_prompt = os.getenv("H2Q_NO_PROMPT", "0") == "1"
        if sys.stdin.isatty() and not no_prompt:
            input("\n程序已停止。按回车键关闭窗口...")

if __name__ == "__main__":
    train_rolling_system()