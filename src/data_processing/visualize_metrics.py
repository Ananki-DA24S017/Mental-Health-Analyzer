# src/data_processing/visualize_metrics.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def visualize_pipeline_metrics(metrics_file):
    """
    Create visualizations of pipeline throughput and speed metrics
    """
    if metrics_file.endswith('.json'):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:  # CSV file
        metrics = pd.read_csv(metrics_file).to_dict(orient='records')[0]
    
    # Create output directory for visualizations
    os.makedirs('reports', exist_ok=True)
    
    # Plot processing times
    plt.figure(figsize=(10, 6))
    steps = ['load', 'clean', 'split']
    times = [metrics[f'{step}_time'] for step in steps]
    
    plt.bar(steps, times)
    plt.title('Processing Time by Pipeline Stage')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Pipeline Stage')
    plt.savefig('reports/processing_times.png')
    
    # Plot throughput
    plt.figure(figsize=(10, 6))
    throughputs = [metrics[f'{step}_throughput'] for step in steps]
    throughputs.append(metrics['overall_throughput'])
    
    plt.bar(steps + ['overall'], throughputs)
    plt.title('Processing Throughput by Pipeline Stage')
    plt.ylabel('Throughput (records/second)')
    plt.xlabel('Pipeline Stage')
    plt.savefig('reports/processing_throughput.png')
    
    print("Visualizations created in 'reports' directory")

if __name__ == "__main__":
    metrics_file = "data/processed/pipeline_metrics.json"
    visualize_pipeline_metrics(metrics_file)