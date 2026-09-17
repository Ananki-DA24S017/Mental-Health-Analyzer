# src/model_training/visualize_models.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def visualize_model_comparison(comparison_file, output_dir):
    """Create visualizations of model performance"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model comparison data
    df = pd.read_csv(comparison_file)
    
    # Sort by test accuracy
    df = df.sort_values('test_accuracy', ascending=False)
    
    # Plot test accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='test_accuracy', data=df)
    plt.title('Model Comparison - Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_accuracy_comparison.png")
    
    # Plot F1 score comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='test_f1', data=df)
    plt.title('Model Comparison - Test F1 Score')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_f1_comparison.png")
    
    # Plot training time comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='train_time', data=df)
    plt.title('Model Comparison - Training Time')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_training_time_comparison.png")
    
    # Create a scatter plot of accuracy vs. training time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='train_time', y='test_accuracy', hue='model_name', size='test_f1', 
                   sizes=(100, 200), data=df)
    plt.title('Accuracy vs. Training Time')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_time.png")
    
    print(f"Model visualization created in {output_dir}")

if __name__ == "__main__":
    comparison_file = "models/model_comparison.csv"
    output_dir = "reports/model_comparison"
    visualize_model_comparison(comparison_file, output_dir)