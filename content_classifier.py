#!/usr/bin/env python3
"""
Weeg Content Type Classifier - Prediction Tool

This script provides a command-line interface for predicting the content type of a website
using a pre-trained Random Forest model. It's part of the Weeg platform's content processing pipeline.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, Any, Optional, List

import numpy as np
import requests
from bs4 import BeautifulSoup
from goose3 import Goose
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


class ContentTypeClassifier:
    """A class for predicting the content type of a website."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the classifier with paths to model artifacts.
        
        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = models_dir
        self.model = None
        self.label_encoder = None
        self.tfidf_vectorizers = None
        self.sentence_model = None
        self.tags = [
            'div', 'p', 'a', 'button', 'input', 'form', 'img', 'li', 'ul', 'ol',
            'h1', 'h2', 'h3', 'section', 'article', 'header', 'footer', 'nav',
            'video', 'audio', 'canvas'
        ]
        self.sentence_bert_model_name = 'all-MiniLM-L6-v2'
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all required models from disk."""
        try:
            # Define model paths
            model_path = os.path.join(self.models_dir, 'random_forest_model.pkl')
            label_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            tfidf_vectorizers_path = os.path.join(self.models_dir, 'tfidf_vectorizers.pkl')
            
            # Check if model files exist
            for path in [model_path, label_encoder_path, tfidf_vectorizers_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load models
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            with open(label_encoder_path, 'rb') as le_file:
                self.label_encoder = pickle.load(le_file)
            with open(tfidf_vectorizers_path, 'rb') as tfidf_file:
                self.tfidf_vectorizers = pickle.load(tfidf_file)
            
            # Load Sentence-BERT
            self.sentence_model = SentenceTransformer(self.sentence_bert_model_name)
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            sys.exit(1)
    
    def _extract_html_features(self, html_content: str) -> List[int]:
        """
        Extract HTML tag features from the content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of feature values
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tag_counts = [len(soup.find_all(tag)) for tag in self.tags]
        text_length = len(soup.get_text())
        html_length = len(html_content)
        text_html_ratio = text_length / html_length if html_length else 0
        return tag_counts + [text_html_ratio]
    
    def _extract_goose_data(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata and content from a URL using Goose.
        
        Args:
            url: Website URL
            
        Returns:
            Dictionary with extracted data or None if extraction failed
        """
        try:
            # Initialize Goose for metadata extraction
            g = Goose()
            article = g.extract(url=url)
            
            if not article:
                return None
            
            # Extract Goose data
            goose_data = {
                'title': article.title,
                'description': article.meta_description,
                'keywords': article.meta_keywords,
                'cleaned_text': article.cleaned_text
            }
            
            # Fetch raw HTML for structural features
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            goose_data['html_content'] = response.text
            return goose_data
            
        except Exception as e:
            print(f"Error extracting data from URL: {str(e)}")
            return None
    
    def predict(self, url: str) -> Dict[str, Any]:
        """
        Predict the content type of a website.
        
        Args:
            url: Website URL
            
        Returns:
            Dictionary with prediction results and metadata
        """
        # Extract data from URL
        goose_data = self._extract_goose_data(url)
        if not goose_data:
            return {"error": "Failed to retrieve website content"}
        
        try:
            # Extract HTML features
            html_features = self._extract_html_features(goose_data['html_content'])
            
            # Vectorize text fields
            title_vector = self.tfidf_vectorizers['title'].transform([goose_data['title'] or ''])
            description_vector = self.tfidf_vectorizers['description'].transform([goose_data['description'] or ''])
            text_vector = self.tfidf_vectorizers['cleaned_text'].transform([goose_data['cleaned_text'] or ''])
            
            # Embed title and description using Sentence-BERT
            title_embedding = self.sentence_model.encode(
                goose_data['title'] or '', 
                show_progress_bar=False
            ).reshape(1, -1)
            description_embedding = self.sentence_model.encode(
                goose_data['description'] or '', 
                show_progress_bar=False
            ).reshape(1, -1)
            
            # Combine features
            combined_features = hstack([
                title_vector, 
                description_vector, 
                text_vector, 
                np.array(html_features).reshape(1, -1), 
                title_embedding, 
                description_embedding
            ])
            
            # Pad with zeros if fewer features (if model expects more features)
            expected_features = self.model.n_features_in_
            if combined_features.shape[1] < expected_features:
                padding = np.zeros((1, expected_features - combined_features.shape[1]))
                combined_features = hstack([combined_features, padding])
            
            # Predict category
            predicted_label = self.model.predict(combined_features)
            predicted_category = self.label_encoder.inverse_transform(predicted_label)[0]
            
            # Get prediction probabilities
            prediction_probs = self.model.predict_proba(combined_features)[0]
            top_categories = []
            for i, prob in enumerate(prediction_probs):
                category = self.label_encoder.inverse_transform([i])[0]
                top_categories.append((category, prob))
            
            # Sort by probability (descending)
            top_categories.sort(key=lambda x: x[1], reverse=True)
            
            # Return results
            return {
                'url': url,
                'predicted_category': predicted_category,
                'top_categories': top_categories[:3],  # Top 3 categories with probabilities
                'metadata': {
                    'title': goose_data['title'],
                    'description': goose_data['description'],
                    'keywords': goose_data['keywords']
                }
            }
            
        except Exception as e:
            return {"error": f"Error during prediction: {str(e)}"}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict the content type of a website using a pre-trained model.'
    )
    parser.add_argument(
        'url', 
        type=str, 
        help='URL of the website to classify'
    )
    parser.add_argument(
        '--models-dir', 
        type=str, 
        default='models',
        help='Directory containing model artifacts (default: models)'
    )
    parser.add_argument(
        '--output-format', 
        choices=['text', 'json'], 
        default='text',
        help='Output format (default: text)'
    )
    return parser.parse_args()


def main():
    """Main function to run the classifier."""
    args = parse_arguments()
    
    # Initialize classifier
    classifier = ContentTypeClassifier(models_dir=args.models_dir)
    
    # Make prediction
    result = classifier.predict(args.url)
    
    # Handle errors
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Output results
    if args.output_format == 'json':
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"\nURL: {result['url']}")
        print(f"Predicted Category: {result['predicted_category']}")
        print("\nTop Categories:")
        for category, probability in result['top_categories']:
            print(f"  - {category}: {probability:.4f}")
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            if value:
                print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()


