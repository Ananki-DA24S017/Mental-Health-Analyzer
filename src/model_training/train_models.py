# src/model_training/train_models.py
import pandas as pd
import numpy as np
import os
import time
import json
import logging
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/model_training_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_training")

def load_data(data_dir):
    """Load the preprocessed datasets"""
    try:
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        val_df = pd.read_csv(f"{data_dir}/val.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        # Ensure no NaN values in the data
        train_df = train_df.fillna('')
        val_df = val_df.fillna('')
        test_df = test_df.fillna('')
        
        # Load category mapping
        with open(f"{data_dir}/category_mapping.json", 'r') as f:
            category_mapping = json.load(f)
        
        # Convert category_mapping keys to integers if they're stored as strings
        category_mapping = {cat: int(idx) if isinstance(idx, str) else idx 
                           for cat, idx in category_mapping.items()}
        
        # Create inverse mapping (code -> category)
        inv_category_mapping = {v: k for k, v in category_mapping.items()}
        
        logger.info(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} samples")
        logger.info(f"Categories: {category_mapping}")
        
        return train_df, val_df, test_df, category_mapping, inv_category_mapping
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(train_df, val_df, test_df):
    """Prepare features and targets for model training"""
    X_train = train_df['statement'].values
    y_train = train_df['status_code'].values
    
    X_val = val_df['statement'].values
    y_val = val_df['status_code'].values
    
    X_test = test_df['statement'].values
    y_test = test_df['status_code'].values
    
    logger.info(f"Prepared data: X_train={X_train.shape}, y_train={y_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y_true, inv_category_mapping):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate classification report
    target_names = [inv_category_mapping[i] for i in sorted(set(y_true))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics, report, cm

def train_models(data_dir, models_dir):
    """Train multiple models and track with MLflow"""
    
    # Load data
    train_df, val_df, test_df, category_mapping, inv_category_mapping = load_data(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Configure MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mental-health-analysis")
    
    # Define models to train
    models = {
        "MultinomialNB_TFIDF": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ]),
        "MultinomialNB_CountVec": Pipeline([
            ('vectorizer', CountVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ]),
        "LogisticRegression": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "LinearSVC": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', LinearSVC(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', XGBClassifier(random_state=42))
        ])
    }
    
    # Train each model and track with MLflow
    best_model = None
    best_accuracy = 0
    best_model_name = None
    
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Training model: {model_name}")
        
        try:
            with mlflow.start_run(run_name=model_name) as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run ID: {run_id}")
                
                # Log parameters
                vectorizer = model.named_steps['vectorizer']
                classifier = model.named_steps['classifier']
                
                # Log vectorizer parameters
                mlflow.log_param("vectorizer_type", type(vectorizer).__name__)
                mlflow.log_param("max_features", vectorizer.max_features)
                
                # Log classifier parameters
                mlflow.log_param("classifier_type", type(classifier).__name__)
                if hasattr(classifier, 'alpha'):
                    mlflow.log_param("alpha", classifier.alpha)
                if hasattr(classifier, 'C'):
                    mlflow.log_param("C", classifier.C)
                if hasattr(classifier, 'max_iter'):
                    mlflow.log_param("max_iter", classifier.max_iter)
                if hasattr(classifier, 'n_estimators'):
                    mlflow.log_param("n_estimators", classifier.n_estimators)
                
                # Train the model
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Log training time
                mlflow.log_metric("train_time", train_time)
                logger.info(f"{model_name} trained in {train_time:.2f} seconds")
                
                # Evaluate on validation set
                val_metrics, val_report, val_cm = evaluate_model(model, X_val, y_val, inv_category_mapping)
                
                # Log validation metrics
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
                
                # Evaluate on test set
                test_metrics, test_report, test_cm = evaluate_model(model, X_test, y_test, inv_category_mapping)
                
                # Log test metrics
                for metric_name, metric_value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                # Save confusion matrix as artifact
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=[inv_category_mapping[i] for i in range(len(inv_category_mapping))],
                           yticklabels=[inv_category_mapping[i] for i in range(len(inv_category_mapping))])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.tight_layout()
                
                # Save confusion matrix
                cm_path = f"{models_dir}/{model_name}_confusion_matrix.png"
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path)
                
                # Save the model
                model_path = f"{models_dir}/{model_name}.pkl"
                import joblib
                joblib.dump(model, model_path)
                mlflow.sklearn.log_model(model, model_name)
                
                # Check if this is the best model so far
                if test_metrics['accuracy'] > best_accuracy:
                    best_accuracy = test_metrics['accuracy']
                    best_model = model
                    best_model_name = model_name
                
                # Store results for comparison
                result = {
                    'model_name': model_name,
                    'val_accuracy': val_metrics['accuracy'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1_score'],
                    'train_time': train_time,
                    'run_id': run_id
                }
                results.append(result)
                # Log the results before creating the DataFrame
                print("Results:", results)
                logger.info(f"{model_name} - Test accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
                
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
    
    # Save best model
    if best_model is not None:
        best_model_path = f"{models_dir}/best_model.pkl"
        joblib.dump(best_model, best_model_path)
        
        # Create model info file
        model_info = {
            'model_name': best_model_name,
            'accuracy': best_accuracy,
            'path': best_model_path,
            'category_mapping': category_mapping
        }
        
        with open(f"{models_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Create a comparison DataFrame and save it
    results_df = pd.DataFrame(results, columns=['model_name', 'train_accuracy', 'test_accuracy', 'test_f1', 'train_time'])
    # Log the DataFrame structure
    logger.info(f"Results DataFrame:\n{results_df}")
    # print("Results DataFrame:", results_df)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    results_df.to_csv(f"{models_dir}/model_comparison.csv", index=False)
    
    return best_model_name, best_accuracy, results_df

if __name__ == "__main__":
    data_dir = "data/processed"
    models_dir = "models"
    
    best_model_name, best_accuracy, results_df = train_models(data_dir, models_dir)
    
    print("\nModel Comparison:")
    print(results_df[['model_name', 'test_accuracy', 'test_f1', 'train_time']].to_string(index=False))
    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")