#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete BERT Cookie Button Classifier - FIXED VERSION
- Fixed deprecated parameter names
- Better epoch recommendations to prevent overfitting
- Improved training parameters
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET CLASS
# ============================================================================

class CookieDataset(Dataset):
    """Custom Dataset for cookie button classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# MAIN CLASSIFIER CLASS
# ============================================================================

class CookieClassifier:
    """Complete BERT-based Cookie Button Classifier"""
    
    def __init__(self, model_name='bert-base-german-cased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Label configuration
        self.label_mapping = {
            'PRIVACY_FRIENDLY': 0,  # Reject, essential only, privacy-friendly
            'NEUTRAL': 1,           # Settings, manage, information
            'PRIVACY_RISKY': 2      # Accept all, tracking, dark patterns
        }
        self.id2label = {v: k for k, v in self.label_mapping.items()}
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"cookie_classifier_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🤖 Cookie Classifier initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Model: {self.model_name}")
        print(f"🏷️ Labels: {list(self.label_mapping.keys())}")

    def load_data(self, csv_file):
        """Load and prepare data from CSV file"""
        print(f"\n📂 Loading data from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')
        
        print(f"📊 Loaded {len(df)} samples")
        
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")
        
        # Clean data
        df = df.dropna(subset=['text', 'label']).copy()
        df = df[df['text'].str.len() > 0].copy()
        
        # Filter to known labels
        valid_labels = set(self.label_mapping.keys())
        df = df[df['label'].isin(valid_labels)].copy()
        
        print(f"✅ Valid data: {len(df)} samples")
        
        # Show distribution
        print(f"\n📈 Label distribution:")
        for label, count in df['label'].value_counts().items():
            percentage = count/len(df)*100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Convert labels to numeric
        df['label_id'] = df['label'].map(self.label_mapping)
        
        return df

    def prepare_datasets(self, df, test_size=0.2, val_size=0.1, balance_data=True):
        """Prepare train/val/test datasets"""
        print(f"\n🔧 Preparing datasets...")
        
        # Balance data if requested
        if balance_data:
            df = self._balance_dataset(df)
        
        # Extract texts and labels
        texts = df['text'].values
        labels = df['label_id'].values
        
        # Create stratified splits
        # First: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Second: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        print(f"📊 Dataset splits:")
        print(f"   Training: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"   Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _balance_dataset(self, df, max_samples_per_class=200):
        """Balance dataset by undersampling majority classes"""
        print(f"⚖️ Balancing dataset (max {max_samples_per_class} per class)...")
        
        label_counts = df['label'].value_counts()
        target_size = min(label_counts.min(), max_samples_per_class)
        
        balanced_dfs = []
        for label in label_counts.index:
            label_df = df[df['label'] == label].copy()
            if len(label_df) > target_size:
                label_df = label_df.sample(n=target_size, random_state=42)
            balanced_dfs.append(label_df)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Balanced to {len(df_balanced)} samples")
        for label, count in df_balanced['label'].value_counts().items():
            print(f"   {label}: {count}")
        
        return df_balanced

    def setup_model(self):
        """Initialize model and tokenizer"""
        print(f"\n🤖 Setting up model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_mapping),
            id2label=self.id2label,
            label2id=self.label_mapping
        )
        
        print(f"✅ Model and tokenizer ready")

    def train(self, train_data, val_data, num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the model with conservative defaults to prevent overfitting"""
        print(f"\n🏋️ Training model...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create datasets
        train_dataset = CookieDataset(X_train, y_train, self.tokenizer)
        val_dataset = CookieDataset(X_val, y_val, self.tokenizer)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        
        print(f"📊 Training configuration:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Class weights: {class_weights}")
        
        # FIXED: Updated parameter names for newer transformers versions
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",  # FIXED: was "evaluation_strategy"
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            warmup_steps=50,  # Reduced warmup for smaller dataset
            dataloader_num_workers=0,  # Avoid Windows issues
            report_to=None,  # Disable wandb
            seed=42  # For reproducibility
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Reduced patience
        )
        
        # Train
        print("🚀 Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✅ Training completed!")
        print(f"📁 Model saved to: {self.output_dir}")
        
        return train_result

    def _compute_metrics(self, eval_pred):
        """Compute metrics during training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}

    def evaluate(self, test_data):
        """Comprehensive evaluation on test data"""
        print(f"\n📊 Evaluating model...")
        
        X_test, y_test = test_data
        test_dataset = CookieDataset(X_test, y_test, self.tokenizer)
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Test Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Detailed report
        class_names = [self.id2label[i] for i in sorted(self.id2label.keys())]
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(f"\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, class_names)
        
        # Save results
        self._save_evaluation_results(accuracy, report, cm, X_test, y_test, y_pred)
        
        return accuracy, y_pred

    def _plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Cookie Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved")

    def _save_evaluation_results(self, accuracy, report, cm, X_test, y_test, y_pred):
        """Save evaluation results"""
        # Save metrics
        results = {
            'accuracy': float(accuracy),
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'label_mapping': self.label_mapping,
            'test_samples': len(X_test)
        }
        
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed report
        with open(f"{self.output_dir}/classification_report.txt", 'w') as f:
            f.write(report)
        
        # Save predictions
        results_df = pd.DataFrame({
            'text': X_test,
            'true_label': [self.id2label[label] for label in y_test],
            'predicted_label': [self.id2label[label] for label in y_pred],
            'correct': y_test == y_pred
        })
        results_df.to_csv(f"{self.output_dir}/predictions.csv", index=False, encoding='utf-8')
        
        # Save misclassifications
        errors = results_df[~results_df['correct']].copy()
        if len(errors) > 0:
            errors.to_csv(f"{self.output_dir}/errors.csv", index=False, encoding='utf-8')
            print(f"💡 {len(errors)} misclassifications saved for analysis")

    def predict(self, texts):
        """Predict labels for new texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs).item()
                confidence = probs[0][pred_class].item()
            
            predictions.append({
                'text': text,
                'predicted_label': self.id2label[pred_class],
                'confidence': confidence,
                'probabilities': {
                    self.id2label[i]: float(probs[0][i]) 
                    for i in range(len(self.id2label))
                }
            })
        
        return predictions

    def load_trained_model(self, model_path):
        """Load a previously trained model"""
        print(f"📂 Loading trained model from: {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"✅ Model loaded successfully")

# ============================================================================
# UTILITY FUNCTIONS - UPDATED WITH BETTER EPOCH RECOMMENDATIONS
# ============================================================================

def get_training_params(data_size):
    """Get recommended training parameters based on data size - CONSERVATIVE TO PREVENT OVERFITTING"""
    if data_size >= 2000:
        return {'epochs': 3, 'lr': 2e-5, 'batch_size': 16}
    elif data_size >= 1000:
        return {'epochs': 3, 'lr': 2e-5, 'batch_size': 16}  # Still conservative
    elif data_size >= 500:
        return {'epochs': 4, 'lr': 3e-5, 'batch_size': 16}  # Reduced from 5 to 4
    elif data_size >= 200:
        return {'epochs': 4, 'lr': 3e-5, 'batch_size': 8}   # Reduced from 8 to 4!
    else:
        return {'epochs': 5, 'lr': 5e-5, 'batch_size': 4}   # Reduced from 10 to 5

def create_example_data():
    """Create example data for testing"""
    print("📝 Creating example data...")
    
    examples = [
        # PRIVACY_FRIENDLY
        {"text": "Alle Cookies ablehnen", "label": "PRIVACY_FRIENDLY"},
        {"text": "Nur notwendige Cookies", "label": "PRIVACY_FRIENDLY"},
        {"text": "Tracking verweigern", "label": "PRIVACY_FRIENDLY"},
        {"text": "Werbefrei nutzen", "label": "PRIVACY_FRIENDLY"},
        
        # NEUTRAL
        {"text": "Cookie-Einstellungen", "label": "NEUTRAL"},
        {"text": "Datenschutz verwalten", "label": "NEUTRAL"},
        {"text": "Mehr Informationen", "label": "NEUTRAL"},
        {"text": "Präferenzen anpassen", "label": "NEUTRAL"},
        
        # PRIVACY_RISKY
        {"text": "Alle Cookies akzeptieren", "label": "PRIVACY_RISKY"},
        {"text": "OK", "label": "PRIVACY_RISKY"},
        {"text": "Weiter", "label": "PRIVACY_RISKY"},
        {"text": "Einverstanden", "label": "PRIVACY_RISKY"}
    ]
    
    df = pd.DataFrame(examples)
    df.to_csv("example_cookie_data.csv", index=False, encoding='utf-8')
    print(f"✅ Created example data: example_cookie_data.csv")
    
    return df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function"""
    print("🚀 BERT Cookie Button Classifier - FIXED VERSION")
    print("=" * 60)
    
    # Configuration
    CSV_FILE = "formatted_cookie_data.csv"  # Update this!
    MODEL_NAME = "bert-base-german-cased"  # German BERT for German cookie buttons
    
    # Check if data file exists
    if not os.path.exists(CSV_FILE):
        print(f"❌ Data file not found: {CSV_FILE}")
        print("💡 Options:")
        print("   1. Update CSV_FILE path in the script")
        print("   2. Run the data formatter script first")
        print("   3. Create example data")
        
        choice = input("\nCreate example data? (y/n): ")
        if choice.lower() == 'y':
            create_example_data()
            CSV_FILE = "example_cookie_data.csv"
        else:
            return
    
    try:
        # Initialize classifier
        classifier = CookieClassifier(model_name=MODEL_NAME)
        
        # Load and prepare data
        df = classifier.load_data(CSV_FILE)
        
        if len(df) < 50:
            print("❌ Insufficient data for training (need at least 50 samples)")
            return
        
        # Prepare datasets
        train_data, val_data, test_data = classifier.prepare_datasets(
            df, test_size=0.2, val_size=0.1, balance_data=True
        )
        
        # Setup model
        classifier.setup_model()
        
        # Get CONSERVATIVE training parameters
        params = get_training_params(len(train_data[0]))
        print(f"\n🎯 CONSERVATIVE parameters for {len(train_data[0])} samples:")
        print(f"   Epochs: {params['epochs']} (reduced to prevent overfitting)")
        print(f"   Learning rate: {params['lr']}")
        print(f"   Batch size: {params['batch_size']}")
        print(f"   💡 BERT is pre-trained - fewer epochs are better!")
        
        # Train model
        train_result = classifier.train(
            train_data, val_data,
            num_epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['lr']
        )
        
        # Evaluate model
        accuracy, predictions = classifier.evaluate(test_data)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📁 Model saved to: {classifier.output_dir}")
        print(f"🎯 Final accuracy: {accuracy:.4f}")
        
        # Test with examples
        print(f"\n🧪 Testing with examples:")
        test_texts = [
            "Alle Cookies akzeptieren",
            "Nur notwendige Cookies",
            "Cookie-Einstellungen",
            "OK",
            "Datenschutz verwalten"
        ]
        
        predictions = classifier.predict(test_texts)
        for pred in predictions:
            print(f"   '{pred['text']}' -> {pred['predicted_label']} ({pred['confidence']:.3f})")
        
        print(f"\n✅ Classifier ready for use!")
        print(f"💡 Use classifier.predict(['your text']) to classify new cookie buttons")
        print(f"⚠️  Watch for overfitting signs: high train accuracy, low test accuracy")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()