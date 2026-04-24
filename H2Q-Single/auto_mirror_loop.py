import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import random
import pandas as pd
from openai import OpenAI
from secret_config import get_deepseek_api_key

# ==========================================
# 🪞 H2Q 镜像反射系统 (The Mirror Loop)
# ==========================================
# 功能：捕捉模型的"胡言乱语" -> 导师重塑为"金句" -> 注入训练
# 目的：通过纠错反馈（RLHF的雏形）建立正确的语言表达

# 🔥 配置区域 (请修改为您的 API 信息)
API_KEY = get_deepseek_api_key()
BASE_URL = "https://api.deepseek.com"            # API 地址
MODEL_NAME = "deepseek-chat"                     # 导师模型

# 路径配置
CHECKPOINT_PATH = 'h2q_fineweb.pt'               # 读取正在训练的模型
INJECTION_DIR = r"D:\H2Q_Cache_Zone\Injection"   # 注入目标

# 模型超参数 (必须与训练代码完全一致)
CONFIG = {
    'dim': 768,           
    'factor_size': 32,
    'fixed_rank': 8,       
    'depth': 12,           
    'seq_len': 128,        
    'batch_size': 1,       # 推理模式
    'dropout_rate': 0.0,
    'axiom_lambda': 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)

if not os.path.exists(INJECTION_DIR): os.makedirs(INJECTION_DIR)

# ==========================================
# 1. H2Q 模型定义 (必须完全一致以加载权重)
# ==========================================
class WaveStructureBank(nn.Module):
    def __init__(self, num_blocks, rank):
        super().__init__()
        self.sub_blocks = num_blocks // 4; self.rank = rank
        self.factors_A = nn.Parameter(torch.zeros(rank, 4, self.sub_blocks, self.sub_blocks, device=device))
    def get_factors(self): return self.factors_A

class BalancedHamiltonLayer(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.dim, self.factor_size, self.structure_bank = dim, factor_size, structure_bank
        self.factors_B = nn.Parameter(torch.zeros(rank, factor_size, factor_size, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
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
        mask = torch.triu(torch.ones(T,T,device=device)*float('-inf'),1)
        att = F.softmax(att+mask, dim=-1)
        y = (att @ v).transpose(1,2).reshape(B,T,C)
        return self.o_proj(y)

class HolographicReversibleBlock(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.n1 = nn.LayerNorm(dim//2); self.attn = QuaternionAttention(dim//2, factor_size, structure_bank, rank)
        self.n2 = nn.LayerNorm(dim//2); self.f1 = BalancedHamiltonLayer(dim//2, factor_size, structure_bank, rank)
        self.act = nn.GELU(); self.f2 = BalancedHamiltonLayer(dim//2, factor_size, structure_bank, rank)
    def f(self, x): return self.attn(self.n1(x))
    def g(self, x): return self.f2(self.act(self.f1(self.n2(x))))
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1,y2], dim=-1)

class H2Q_Transformer(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.bank = WaveStructureBank(config['dim']//2//config['factor_size'], config['fixed_rank'])
        self.emb = nn.Embedding(vocab_size, config['dim'])
        self.pos = nn.Parameter(torch.randn(1, config['seq_len'], config['dim'])*0.02)
        self.layers = nn.ModuleList([HolographicReversibleBlock(config['dim'], config['factor_size'], self.bank, config['fixed_rank']) for _ in range(config['depth'])])
        self.head = nn.Linear(config['dim'], vocab_size)
    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.size(1), :]
        for l in self.layers: x = l(x)
        return self.head(x)

# ==========================================
# 2. 核心逻辑：加载、生成、修正、注入
# ==========================================
def sanitize_state_dict(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."): new_dict[k[10:]] = v
        else: new_dict[k] = v
    return new_dict

def load_latest_model():
    """加载最新的训练权重"""
    print("🔄 同步 H2Q 大脑状态...", end="")
    try:
        model = H2Q_Transformer(256, CONFIG).to(device)
        model.eval()
        for _ in range(5):
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
                model.load_state_dict(sanitize_state_dict(ckpt['model']))
                print(f" 成功! (Step: {ckpt.get('chunk_counter', '?')})")
                return model
            except:
                time.sleep(1)
        print("❌ 加载超时")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def generate_raw_thought(model):
    """诱导 H2Q 产生原始想法 (可能是乱码)"""
    # 诱导词：包含自我、世界、逻辑
    seeds = ["I think", "The world is", "Logic is", "I am", "Why", "If"]
    seed = random.choice(seeds)
    
    input_bytes = list(seed.encode('utf-8'))
    x = torch.tensor([input_bytes], dtype=torch.long, device=device)
    generated = []
    
    with torch.no_grad():
        # 生成 100 个字符，或者直到句号
        for _ in range(100): 
            cond = x[:, -CONFIG['seq_len']:]
            logits = model(cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 0: break
            
            x = torch.cat((x, torch.tensor([[next_token]], device=device)), dim=1)
            generated.append(next_token)
            
            # 遇到句号或换行符停止，保证是一句话
            if next_token == ord('.') or next_token == ord('\n'):
                break
            
    try:
        raw_text = seed + bytes(generated).decode('utf-8', errors='ignore')
        return raw_text.strip()
    except: return None

def mirror_correction(raw_text):
    """镜像修正：让导师把乱语变成真理"""
    print(f"🗣️ H2Q 原话: \033[93m{raw_text}\033[0m")
    
    if len(raw_text) < 5: return None

    system_prompt = """
    You are a "Mirror Intelligence". 
    Your student (a young AI) is trying to speak but has broken grammar and logic.
    Your task:
    1. Guess what the student *intended* to say based on the keywords.
    2. Rewrite it into a PERFECT, profound, and logical sentence.
    3. If the input is total gibberish, generate a profound philosophical statement starting with the same word.
    
    Output ONLY the corrected sentence. No explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Student said: '{raw_text}'"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        corrected = response.choices[0].message.content.strip()
        print(f"🪞 镜像修正: \033[92m{corrected}\033[0m")
        return corrected
    except Exception as e:
        print(f"⚠️ 镜像破碎 (API Error): {e}")
        return None

def inject_correction(raw_input, corrected_output):
    """
    关键策略：
    我们不把 H2Q 的乱语作为 Input，而是把 '修正后的句子' 直接作为训练数据。
    这告诉模型："当你本来想说 X 时，你应该说 Y。"
    """
    # 构造数据：直接注入完美句子，或者构造 User/AI 对话
    # 这里选择直接注入完美句子，强化语言模型能力
    text = f"{corrected_output}\n<|endoftext|>"
    
    fname = f"mirror_{int(time.time())}_{random.randint(0,999)}.parquet"
    fpath = os.path.join(INJECTION_DIR, fname)
    try:
        df = pd.DataFrame({'text': [text]})
        df.to_parquet(fpath)
        print(f"💉 修正已注入: {fname}")
    except: pass

# ==========================================
# 3. 主循环
# ==========================================
if __name__ == "__main__":
    print("=========================================")
    print("   H2Q 镜像反射系统 (Mirror System)")
    print("   [胡言乱语] -> [完美重述] -> [注入]")
    print("=========================================")
    
    while True:
        model = load_latest_model()
        if model:
            # 1. H2Q 尝试说话
            raw = generate_raw_thought(model)
            if raw:
                # 2. 导师修正
                corrected = mirror_correction(raw)
                if corrected:
                    # 3. 注入完美版本
                    inject_correction(raw, corrected)
        
        print("⏳ 冷却 (30s)...")
        time.sleep(30)