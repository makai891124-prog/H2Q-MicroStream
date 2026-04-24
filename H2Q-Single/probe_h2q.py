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
# 🧠 H2Q 苏格拉底闭环 (V2.0 - 长文本增强版)
# ==========================================

# 🔥 配置区域
API_KEY = get_deepseek_api_key()
BASE_URL = "https://api.deepseek.com"            # API 地址
MODEL_NAME = "deepseek-chat"                     # 导师模型

# 路径配置
CHECKPOINT_PATH = 'h2q_fineweb.pt'               
INJECTION_DIR = r"D:\H2Q_Cache_Zone\Injection"   

# 模型超参数 (保持一致)
CONFIG = {
    'dim': 768, 'factor_size': 32, 'fixed_rank': 8, 'depth': 12,           
    'seq_len': 128, 'batch_size': 1, 'dropout_rate': 0.0, 'axiom_lambda': 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

if not os.path.exists(INJECTION_DIR): os.makedirs(INJECTION_DIR)

# ... [此处省略模型类定义 WaveStructureBank 等 5 个类] ...
# ... [请务必保留之前的类定义，为了节省篇幅这里不重复粘贴] ...
# ... [如果您没有保存类定义，请从之前的回复或训练脚本中复制] ...

# (为了代码可运行，这里必须要有类定义。如果您直接覆盖文件，请确保类定义还在)
# 建议：直接在之前的 auto_socratic_loop.py 基础上，只替换下面的函数部分。
# 或者，为了方便，我这里提供完整的类定义占位符，请您确保代码里有这些类。

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
# 2. 核心逻辑 (已优化)
# ==========================================
def sanitize_state_dict(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."): new_dict[k[10:]] = v
        else: new_dict[k] = v
    return new_dict

def load_latest_model():
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

def generate_inquiry(model):
    """诱导 H2Q 提出一个问题 (长文本版)"""
    seeds = [
        "Question: What is",
        "I am confused about",
        "Explain the concept of",
        "Why does",
        "How to calculate"
    ]
    seed = random.choice(seeds)
    
    input_bytes = list(seed.encode('utf-8'))
    x = torch.tensor([input_bytes], dtype=torch.long, device=device)
    
    generated = []
    
    with torch.no_grad():
        # 🔥 优化：允许生成 200 个字符，给它足够的空间把话说完
        for _ in range(200): 
            cond = x[:, -CONFIG['seq_len']:]
            logits = model(cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 0: break 
            
            x = torch.cat((x, torch.tensor([[next_token]], device=device)), dim=1)
            generated.append(next_token)
            
            # 🔥 优化：遇到问号或换行符就停止，保证句子完整性
            if next_token == ord('?') or next_token == ord('\n'):
                break
    
    try:
        full_text = seed + bytes(generated).decode('utf-8', errors='ignore')
        # 清理：只取第一行，且确保有问号
        question = full_text.split('\n')[0].strip()
        if not question.endswith('?'):
            question += "?" # 帮它补上问号
        return question
    except:
        return None

def ask_teacher(question):
    """向 DeepSeek 老师请教"""
    print(f"🤔 H2Q 提问: \033[94m{question}\033[0m")
    
    if len(question) < 5: return None # 太短了
        
    try:
        # 🔥 优化：告诉老师学生还在学习语言，要宽容
        system_prompt = """
        You are a helpful tutor answering a young AI. 
        The AI is still learning English grammar, so its questions might be broken or contain typos.
        Please try to understand the INTENT of the question and provide a clear, logical answer.
        Format: Thought: [Analysis] Answer: [Explanation]
        """
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ 导师掉线: {e}")
        return None

def inject_knowledge(question, answer):
    text = f"User: {question}\nAI: {answer}\n<|endoftext|>"
    fname = f"socratic_{int(time.time())}_{random.randint(0,999)}.parquet"
    fpath = os.path.join(INJECTION_DIR, fname)
    try:
        df = pd.DataFrame({'text': [text]})
        df.to_parquet(fpath)
        print(f"💉 \033[92m知识已注入: {fname}\033[0m")
    except:
        print("❌ 注入写入失败")

# ==========================================
# 3. 主循环
# ==========================================
if __name__ == "__main__":
    print("=========================================")
    print("   H2Q 苏格拉底闭环 (V2.0 - 宽容模式)")
    print("   [H2Q 提问] -> [DeepSeek 解答] -> [注入]")
    print("=========================================")
    
    while True:
        model = load_latest_model()
        if model:
            question = generate_inquiry(model)
            if question:
                answer = ask_teacher(question)
                if answer:
                    inject_knowledge(question, answer)
        
        print("⏳ 消化中 (60s)...")
        time.sleep(60)