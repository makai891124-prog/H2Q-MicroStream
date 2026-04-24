import pandas as pd
import numpy as np
import os

def simple_linregress(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n < 2: return 0
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x*x)
    sum_xy = np.sum(x*y)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x**2
    if denominator == 0: return 0
    return numerator / denominator

files = [
    "evolution_telemetry_7x24.csv",
    "evolution_telemetry.csv",
    "evolution_telemetry_24h.csv",
    "evolution_telemetry_cycle_1.csv",
    "evolution_telemetry_open.csv"
]

results = []

for f in files:
    if not os.path.exists(f):
        continue
    
    try:
        df = pd.read_csv(f)
        if df.empty:
            continue
            
        row_count = len(df)
        t_min, t_max = df['T_Step'].min(), df['T_Step'].max()
        
        causal_start = df['Causal_Loss_EMA'].iloc[0]
        causal_end = df['Causal_Loss_EMA'].iloc[-1]
        causal_pct_change = (causal_end - causal_start) / causal_start * 100 if causal_start != 0 else 0
        
        # Linear slope full
        slope_full = simple_linregress(df['T_Step'], df['Causal_Loss_EMA'])
        
        # Last 20%
        last_20_idx = int(row_count * 0.8)
        df_last20 = df.iloc[last_20_idx:]
        slope_last20 = simple_linregress(df_last20['T_Step'], df_last20['Causal_Loss_EMA'])
        
        sparsity_mean = df_last20['Topology_Sparsity'].mean()
        sparsity_std = df_last20['Topology_Sparsity'].std()
        
        plateau = abs(slope_last20) < 1e-4
        emergence_proxy = (slope_last20 < 0) and (sparsity_mean > 0.5) and (sparsity_std < 0.2)
        
        results.append({
            "File": f,
            "Rows": row_count,
            "T_Step": f"{t_min}/{t_max}",
            "Causal_EMA_Start": round(causal_start, 6),
            "Causal_EMA_End": round(causal_end, 6),
            "Causal_Pct": f"{round(causal_pct_change, 2)}%",
            "Slope_Full": f"{slope_full:.2e}",
            "Slope_Last20": f"{slope_last20:.2e}",
            "Sparsity_M": round(sparsity_mean, 4),
            "Sparsity_S": round(sparsity_std, 4),
            "Plateau": plateau,
            "Emergence": emergence_proxy
        })
    except Exception as e:
        print(f"Error processing {f}: {e}")

print(pd.DataFrame(results).to_string(index=False))
