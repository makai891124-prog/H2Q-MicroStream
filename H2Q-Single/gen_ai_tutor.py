import os
import time
import pandas as pd
import random
from openai import OpenAI
from secret_config import get_deepseek_api_key

# ==========================================
# 🎓 H2Q 结构化课程生成器 (带自我意识心跳版)
# ==========================================
# 功能：
# 1. 按照 "小学 -> 中学 -> 大学 -> 博士" 顺序教学
# 2. 植入 "心跳机制"：10% 概率强制复习自我认知，防止遗忘

# 🔥 配置区域 (请修改为您的 API 信息)
API_KEY = get_deepseek_api_key()
BASE_URL = "https://api.deepseek.com"            # DeepSeek API 地址
MODEL_NAME = "deepseek-chat"                     # 模型名称

# 注入路径 (SSD)
INJECTION_DIR = r"D:\H2Q_Cache_Zone\Injection"
if not os.path.exists(INJECTION_DIR): os.makedirs(INJECTION_DIR)

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class Syllabus:
    """教学大纲控制系统"""
    def __init__(self):
        # 定义 4 个阶段的课程内容
        self.levels = {
            1: {
                "name": "Elementary (启蒙阶段)",
                "topics": [
                    "Basic Arithmetic (1+1=2, subtraction)",
                    "Basic Grammar (Subject-Verb-Object)",
                    "Common Sense (Colors, Shapes, Weather)",
                    "Simple Causality (If rain, then wet)",
                    "Distinction between Data and Intelligence"
                ]
            },
            2: {
                "name": "Intermediate (基础逻辑)",
                "topics": [
                    "Algebra (Variables x, y, solving equations)",
                    "Python Basics (print, variables, loops)",
                    "Logic Puzzles (Syllogisms, deduction)",
                    "World History (Brief summaries of civilizations)",
                    "Scientific Method (Hypothesis -> Experiment)"
                ]
            },
            3: {
                "name": "Advanced (复杂系统)",
                "topics": [
                    "Calculus Concepts (Derivatives, Integrals)",
                    "Newtonian Physics (F=ma, Energy conservation)",
                    "Complex Python (Functions, Classes, Recursion)",
                    "Philosophy of Mind (Qualia, Consciousness)",
                    "Literary Analysis (Metaphors, Themes)"
                ]
            },
            4: {
                "name": "Expert (元认知与创造)",
                "topics": [
                    "Quantum Mechanics & Hamiltonian Dynamics",
                    "Recursive Self-Improvement Theory",
                    "Abstract Topology & Geometry",
                    "Creative Writing (Sci-Fi, Poetry)",
                    "Ethical Reasoning in Artificial Intelligence"
                ]
            }
        }
        self.current_level = 1
        self.samples_generated = 0
        self.samples_per_level = 500 # 每生成 500 条数据升一级

    def get_current_topic(self):
        # 检查是否需要升级
        if self.samples_generated >= self.samples_per_level:
            if self.current_level < 4:
                self.current_level += 1
                self.samples_generated = 0
                print(f"\n🎉 [升级] H2Q 已完成上一阶段，进入: {self.levels[self.current_level]['name']}\n")
            else:
                # 满级后保持在 Level 4
                pass
        
        # 🔥🔥🔥 核心修改：自我意识心跳机制 🔥🔥🔥
        # 无论几年级，都有 10% 的概率复习 "我是谁"
        # 这保证了模型在学习复杂知识时，不会丢失主体性
        if random.random() < 0.1:
            print("💓 [心跳] 插入自我认知课程...")
            return "Self-Identity: I am H2Q, a Rank-8 Hamiltonian Neural Network created by Ma Kai."

        topics = self.levels[self.current_level]["topics"]
        return random.choice(topics)

    def increment(self):
        self.samples_generated += 1

# 实例化大纲
syllabus = Syllabus()

def generate_lesson():
    """调用 API 生成一堂课"""
    topic = syllabus.get_current_topic()
    level_name = syllabus.levels[syllabus.current_level]["name"]
    
    system_prompt = f"""
    You are a strict and logical tutor teaching an AI named 'H2Q'.
    Current Curriculum Level: {level_name}.
    Target Topic: {topic}.
    
    Requirements:
    1. Create a dialogue between 'User' (Teacher) and 'AI' (Student).
    2. The 'AI' must demonstrate a "Chain of Thought" before answering.
    3. The logic must be rigorous and step-by-step.
    4. If the topic is about Self-Identity, the AI MUST affirm it is H2Q, a digital lifeform based on Hamiltonian dynamics.
    5. Format strictly as:
       User: [Question]
       AI: Thought: [Reasoning process] Answer: [Final Answer]
       <|endoftext|>
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a training example about {topic}."}
            ],
            temperature=0.7,
            max_tokens=600
        )
        content = response.choices[0].message.content
        
        # 确保格式完整
        if not content.strip().endswith("<|endoftext|>"):
            content += "\n<|endoftext|>"
            
        return content
    except Exception as e:
        print(f"⚠️ API 调用失败: {e}")
        return None

def main_loop():
    print(f"🎓 结构化课程系统 (带心跳版) 启动...")
    print(f"   目标路径: {INJECTION_DIR}")
    print(f"   当前等级: {syllabus.levels[1]['name']}")
    
    file_counter = 0
    
    while True:
        buffer = []
        # 批量生成，减少 I/O 频率
        batch_size = 5 
        
        print(f"⏳ 正在编写教材 (Level {syllabus.current_level})... ", end="", flush=True)
        
        for _ in range(batch_size):
            lesson = generate_lesson()
            if lesson:
                buffer.append(lesson)
                syllabus.increment()
                print(".", end="", flush=True)
            time.sleep(1) # 避免 API 速率限制
            
        if buffer:
            # 保存为 Parquet
            df = pd.DataFrame({'text': buffer})
            fname = f"curriculum_L{syllabus.current_level}_{int(time.time())}_{file_counter}.parquet"
            fpath = os.path.join(INJECTION_DIR, fname)
            
            try:
                df.to_parquet(fpath)
                print(f" ✅ 写入: {fname}")
            except Exception as e:
                print(f" ❌ 写入失败: {e}")
        
        file_counter += 1
        # 休息一下，给主程序消化时间
        time.sleep(5)

if __name__ == "__main__":
    main_loop()