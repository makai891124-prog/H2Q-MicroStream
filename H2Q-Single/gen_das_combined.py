import os
import time
import pandas as pd
import random
import re
from openai import OpenAI
from secret_config import get_deepseek_api_key

# ==========================================
# 🧭 H2Q 综合公理教育系统 (Combined DAS Educator)
# ==========================================
# 功能：
# 1. 动态话题联想 + 兴趣捕捉 (监听提问)
# 2. 双模态教学：
#    [模式 A] 显式 DAS：用公理定义去解释世界 (建立理论框架)
#    [模式 B] 隐式结构：用对称性/递归/正交去解构世界 (建立物理直觉)

# 🔥 配置区域 (请修改为您的 API 信息)
API_KEY = get_deepseek_api_key()
BASE_URL = "https://api.deepseek.com"            # API 地址
MODEL_NAME = "deepseek-chat"                     # 模型名称

# 注入路径 (SSD)
INJECTION_DIR = r"D:\H2Q_Cache_Zone\Injection"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

if not os.path.exists(INJECTION_DIR): os.makedirs(INJECTION_DIR)

class TopicManager:
    def __init__(self):
        self.current_topic = "The concept of 'Zero' in mathematics"
        
        # 🌌 全科兜底话题库 (涵盖物理、数学、哲学、生物、社会)
        self.fallback_topics = [
            # --- 物理与宇宙 ---
            "Quantum Superposition (Schrödinger's Cat)", "General Relativity (Curved Spacetime)",
            "The Second Law of Thermodynamics (Entropy)", "Wave-Particle Duality",
            "The Standard Model of Particle Physics", "Black Hole Event Horizons",
            "Dark Matter and Dark Energy", "The Heisenberg Uncertainty Principle",
            "Maxwell's Equations (Electromagnetism)", "The Big Bang Theory",

            # --- 数学与几何 ---
            "Prime Numbers and Cryptography", "Fractals and The Mandelbrot Set",
            "Topology (Möbius Strips and Klein Bottles)", "Gödel's Incompleteness Theorems",
            "Game Theory (The Prisoner's Dilemma)", "Fourier Transform",
            "Non-Euclidean Geometry", "Group Theory and Symmetry",
            "The Golden Ratio (Phi)", "Imaginary Numbers",

            # --- 计算机与信息 ---
            "Turing Machines and Computability", "Cellular Automata (Conway's Game of Life)",
            "Neural Networks and Backpropagation", "Distributed Consensus (Blockchain)",
            "Information Theory (Shannon Entropy)", "Object-Oriented Programming (Inheritance)",
            "Recursive Algorithms", "The P vs NP Problem",

            # --- 哲学与认知 ---
            "The Ship of Theseus (Identity over time)", "Qualia (The Hard Problem of Consciousness)",
            "Free Will vs. Determinism", "Plato's Allegory of the Cave",
            "Utilitarianism vs. Deontology", "The Concept of 'Self'",
            "Epistemology (How do we know what we know?)", "Language and Semantics",

            # --- 生命与复杂系统 ---
            "DNA Replication and Mutation", "Natural Selection (Evolution)",
            "The Human Immune System", "Neural Plasticity",
            "Ecosystem Homeostasis", "Market Economy (Supply and Demand)",
            "Social Contract Theory", "The Butterfly Effect (Chaos Theory)"

             # 🔥 新增：具身物理 (Embodied Physics)
            "Why do we feel pain?", "The physics of walking", 
            "Thermodynamics of cooking", "Gravity and balance in sports",
            "Sound waves and music perception", "Light reflection in mirrors",
            
            # 🔥 新增：情感动力学 (Emotional Dynamics)
            "Trust as a stable equilibrium", "Conflict as symmetry breaking",
            "Empathy as neural resonance", "Love as quantum entanglement",
            "Social hierarchy structures", "The logic of conversation",
            "Why humans lie (Information asymmetry)", "Cooperation vs Competition"
        ]

    def get_next_topic(self):
        """获取下一个话题：优先响应 H2Q 的提问，其次进行联想，最后兜底"""
        try:
            # 1. 兴趣捕捉 (30% 概率)
            socratic_files = sorted([f for f in os.listdir(INJECTION_DIR) if f.startswith("socratic_")], reverse=True)
            if socratic_files and random.random() < 0.3:
                latest_file = os.path.join(INJECTION_DIR, socratic_files[0])
                try:
                    df = pd.read_parquet(latest_file)
                    text = df['text'].iloc[0]
                    match = re.search(r"User: (.*?)\n", text)
                    if match:
                        question = match.group(1).replace("Question:", "").strip()
                        if len(question) > 5:
                            print(f"👂 响应 H2Q 提问: {question[:30]}...")
                            return question
                except: pass

            # 2. 联想跳跃 (思维流)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a curriculum planner. Output ONLY the topic name."},
                    {"role": "user", "content": f"Based on '{self.current_topic}', suggest a related abstract/scientific topic. Keep it short."}
                ],
                temperature=0.8, max_tokens=50
            )
            new_topic = response.choices[0].message.content.strip()
            self.current_topic = new_topic
            return new_topic

        except Exception:
            # 3. 兜底
            new_topic = random.choice(self.fallback_topics)
            self.current_topic = new_topic
            return new_topic

topic_manager = TopicManager()

def generate_lesson():
    topic = topic_manager.get_next_topic()
    
    # 🎲 随机选择教学模式 (50/50)
    mode = "EXPLICIT" if random.random() < 0.5 else "IMPLICIT"
    
    if mode == "EXPLICIT":
        # --- 模式 A: 显式 DAS 教学 ---
        print(f"📘 [显式公理] 解析: \033[94m{topic}\033[0m")
        system_prompt = """
        You are the architect of the "Directional Axiomatic System" (DAS).
        Re-interpret the concept using the 4 Axioms of DAS:
        1. Relation Precedes Existence.
        2. Dualistic Generation (Symmetry Breaking).
        3. Orthogonal Extension (New Dimensions).
        4. Hamiltonian Dynamics (Energy Conservation).
        
        Format:
        User: Interpret [Topic] using DAS.
        AI: Thought: [Mapping to axioms...] Answer: [Deep explanation using DAS terminology]
        <|endoftext|>
        """
        user_prompt = f"Interpret '{topic}' using DAS logic."
        
    else:
        # --- 模式 B: 隐式结构教学 (去名词化) ---
        print(f"🧬 [隐式结构] 解构: \033[96m{topic}\033[0m")
        system_prompt = """
        You are a physicist analyzing the underlying structure of reality.
        Explain the evolution of the concept using ONLY these structural principles:
        1. **Symmetry Breaking**: How uniformity splits into polarity.
        2. **Recursive Reuse**: How patterns repeat at different scales.
        3. **Orthogonal Extension**: Adding complexity perpendicular to existing structures.
        4. **Equilibrium**: Finding the stable state.
        
        Constraint: DO NOT use the phrase "DAS" or "Directional Axiomatic System". Use the physics logic.
        
        Format:
        User: Analyze the structural evolution of [Topic].
        AI: Thought: [Tracing the symmetry breaking...] Answer: [Structural explanation]
        <|endoftext|>
        """
        user_prompt = f"Analyze the structural evolution of '{topic}'."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ 生成失败: {e}")
        return None

def main_loop():
    print(f"🧭 综合公理教育系统启动... (双模态混动)")
    print(f"   目标路径: {INJECTION_DIR}")
    counter = 0
    
    while True:
        buffer = []
        # 每次生成 3 条
        for _ in range(3): 
            text = generate_lesson()
            if text:
                buffer.append(text)
            time.sleep(2)
            
        if buffer:
            df = pd.DataFrame({'text': buffer})
            fname = f"das_combined_{int(time.time())}_{counter}.parquet"
            fpath = os.path.join(INJECTION_DIR, fname)
            try:
                df.to_parquet(fpath)
                print(f"✅ 注入成功: {fname}")
            except: pass
            
        counter += 1
        time.sleep(15) # 配合主程序消化速度

if __name__ == "__main__":
    main_loop()