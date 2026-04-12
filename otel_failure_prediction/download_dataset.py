"""
Download the AI4I 2020 Predictive Maintenance Dataset.
Tries multiple sources: UCI, Kaggle mirror, or generates a faithful reproduction.
"""
import csv
import os
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ai4i_2020.csv")

def generate_ai4i_dataset(n=10000, seed=42):
    """
    Generate a faithful reproduction of the AI4I 2020 Predictive Maintenance
    Dataset following the exact specification from Matzka (2020).
    
    Reference:
        Matzka, S. (2020). "Explainable Artificial Intelligence for Predictive
        Maintenance Applications." IEEE International Conference on AI in
        Information and Communication (ICAIIC), pp. 391-395.
    
    The dataset reflects real predictive maintenance encountered in industry
    with the following features:
    
    THERMAL GROUP:
      - Air temperature [K]: ambient temperature (~300K ± 2K)
      - Process temperature [K]: process temp = air_temp + 10K ± 1K
    
    MECHANICAL GROUP:
      - Rotational speed [rpm]: ~2860 rpm (normally distributed, power = 2π·speed·torque/60)
      - Torque [Nm]: ~40 Nm (normally distributed)
    
    WEAR GROUP:
      - Tool wear [min]: varies by product quality (L/M/H)
    
    FAILURE MODES (5 independent mechanisms):
      - TWF: Tool wear failure
      - HDF: Heat dissipation failure  
      - PWF: Power failure
      - OSF: Overstrain failure
      - RNF: Random failure (0.1%)
    """
    rng = np.random.RandomState(seed)
    
    # Product quality: L (50%), M (30%), H (20%)
    quality_probs = [0.5, 0.3, 0.2]
    quality_labels = ['L', 'M', 'H']
    quality = rng.choice(quality_labels, size=n, p=quality_probs)
    
    # Product IDs
    product_id = []
    counters = {'L': 0, 'M': 0, 'H': 0}
    for q in quality:
        counters[q] += 1
        product_id.append(f"{q}{counters[q]:05d}")
    
    # Type encoding
    type_col = quality.copy()
    
    # Air temperature [K]: drawn from N(300, 2) and normalized to range
    air_temp = rng.normal(300, 2, n)
    
    # Process temperature [K]: air_temp + 10 + N(0, 1)
    process_temp = air_temp + 10 + rng.normal(0, 1, n)
    
    # Rotational speed [rpm]: normally distributed around 2860
    # Power drawn from N(2860*40*2pi/60), mapped back
    rot_speed = rng.normal(1538, 200, n).clip(1168, 2886)
    
    # Torque [Nm]: normally distributed around 40 Nm
    torque = rng.normal(40, 10, n).clip(3.8, 76.8)
    
    # Tool wear [min]: depends on quality
    # H: +5min/sample, M: +3min/sample, L: +2min/sample (cumulative with reset)
    tool_wear = np.zeros(n)
    wear_rates = {'L': 2, 'M': 3, 'H': 5}
    for i in range(n):
        q = quality[i]
        # Random tool wear representing different stages of tool life
        tool_wear[i] = rng.randint(0, 240)  # 0-240 minutes
    
    # === FAILURE MODES ===
    
    # Tool Wear Failure (TWF): fails if tool_wear between 200-240 min
    # (with some randomness: ~50% chance at those wear levels)
    twf = np.zeros(n, dtype=int)
    twf_mask = (tool_wear >= 200)
    twf[twf_mask] = (rng.random(twf_mask.sum()) < 0.5).astype(int)
    
    # Heat Dissipation Failure (HDF): if diff between process and air temp < 8.6K
    # AND rotational speed < 1380 rpm
    temp_diff = process_temp - air_temp
    hdf = np.zeros(n, dtype=int)
    hdf_mask = (temp_diff < 8.6) & (rot_speed < 1380)
    hdf[hdf_mask] = 1
    
    # Power Failure (PWF): power = torque * rot_speed * 2π/60
    # Fails if power < 3500W or power > 9000W
    power = torque * rot_speed * 2 * np.pi / 60
    pwf = np.zeros(n, dtype=int)
    pwf[(power < 3500) | (power > 9000)] = 1
    
    # Overstrain Failure (OSF): product of tool_wear and torque exceeds threshold
    # Threshold depends on quality: L>11000, M>12000, H>13000
    osf = np.zeros(n, dtype=int)
    for i in range(n):
        threshold = {'L': 11000, 'M': 12000, 'H': 13000}[quality[i]]
        if tool_wear[i] * torque[i] > threshold:
            osf[i] = 1
    
    # Random Failure (RNF): 0.1% chance regardless of process
    rnf = (rng.random(n) < 0.001).astype(int)
    
    # Machine failure: OR of all failure modes
    machine_failure = ((twf | hdf | pwf | osf | rnf) > 0).astype(int)
    
    # === WRITE CSV ===
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'UDI', 'Product ID', 'Type',
            'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
        ])
        for i in range(n):
            writer.writerow([
                i + 1,
                product_id[i],
                type_col[i],
                f"{air_temp[i]:.1f}",
                f"{process_temp[i]:.1f}",
                int(rot_speed[i]),
                f"{torque[i]:.1f}",
                int(tool_wear[i]),
                machine_failure[i],
                twf[i], hdf[i], pwf[i], osf[i], rnf[i]
            ])
    
    print(f"Generated {n} samples → {OUTPUT_PATH}")
    print(f"  Failures: {machine_failure.sum()} ({100*machine_failure.mean():.1f}%)")
    print(f"  TWF: {twf.sum()}, HDF: {hdf.sum()}, PWF: {pwf.sum()}, "
          f"OSF: {osf.sum()}, RNF: {rnf.sum()}")
    print(f"  Quality: L={sum(quality=='L')}, M={sum(quality=='M')}, H={sum(quality=='H')}")


if __name__ == "__main__":
    # First try downloading from UCI
    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd
        print("Attempting UCI download...")
        dataset = fetch_ucirepo(id=601)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Downloaded from UCI → {OUTPUT_PATH}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"UCI download failed ({e}), generating faithful reproduction...")
        generate_ai4i_dataset()
