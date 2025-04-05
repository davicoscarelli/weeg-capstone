# Weeg Content Type Classifier

This project implements a machine learning model that serves as a critical component of the Weeg platform - a decentralized mesh network designed to provide students in underserved regions with access to educational content.

## Project Overview

The Content Type Classifier is a key component in Weeg's web content processing and delivery pipeline. It analyzes web pages to determine their content type (e.g., blog, marketplace, educational site), enabling the system to tailor subsequent processing steps for optimal delivery in low-bandwidth environments.

### Role in Weeg's Architecture

Weeg consists of three main components:
1. **Worker Devices**: Solar-powered, low-cost devices that distribute learning materials over a LoRaWAN mesh network
2. **Gateways**: Local hubs that sync Worker devices with the cloud, reducing the need for constant internet access
3. **Hive Platform**: An AI-driven system that processes student queries, provides tutoring, and optimizes content for low-bandwidth use

This classifier is part of the Hive Platform's content processing pipeline, which:
1. Retrieves web content using Puppeteer
2. **Classifies content types** (this project)
3. Segments pages into semantic roles
4. Simplifies and encodes content for low-bandwidth delivery
5. Optimizes network transmission

## Technical Details

The classifier uses a Random Forest model trained on a dataset of web pages across various content types. Features include:

- **HTML Tag Counts**: Frequencies of specific HTML tags to represent structural composition
- **Text-to-HTML Ratios**: Content density measures
- **TF-IDF Features**: Key textual features from titles, descriptions, and keywords
- **Sentence-BERT Embeddings**: Semantic representations of page titles and descriptions

The model achieved the following performance metrics during cross-validation:
- **Accuracy**: Mean of 82.82% with a standard deviation of 1.2%
- **Precision (Weighted)**: Mean of 84.22%
- **Recall (Weighted)**: Mean of 82.82%
- **F1 Score (Weighted)**: Mean of 82.63%

## Running Inference

The project includes a command-line tool for predicting the content type of websites using the pre-trained model.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

#### Using Setup Scripts (Recommended)

1. Clone this repository:
   ```
   git clone <repository-url>
   cd weeg-capstone
   ```

2. Run the appropriate setup script for your operating system:
   - On macOS/Linux:
     ```
     chmod +x setup.sh
     ./setup.sh
     ```
   - On Windows:
     ```
     setup.bat
     ```

#### Manual Installation

If you prefer to set up manually:

1. Create a virtual environment:
   ```
   python -m venv weeg-env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     weeg-env\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source weeg-env/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Using the Classifier

The classifier is available as a command-line tool in `content_classifier.py`. Here's how to use it:

#### Basic Usage

```bash
python content_classifier.py https://example.com
```

This will output the predicted content type and metadata for the specified URL.

#### Advanced Options

```bash
# Specify a different models directory
python content_classifier.py https://example.com --models-dir /path/to/models

# Get JSON output (useful for programmatic use)
python content_classifier.py https://example.com --output-format json
```

#### Example Output

```
URL: https://example.com
Predicted Category: educational

Top Categories:
  - educational: 0.8234
  - blog: 0.1234
  - news: 0.0532

Metadata:
  - title: Example Domain
  - description: Example website for demonstration purposes
  - keywords: example, domain, demonstration
```

## Training the Model

If you want to train or retrain the model, you can use the Jupyter notebook `training.ipynb`.

### Data Requirements

The notebook expects the following data files in the `data/` directory:
- `website_classification.csv`: Initial dataset with website URLs and categories
- `website_metadata_content.csv`: Generated file with extracted metadata (will be created during training)


### Training Process

The notebook contains the following steps:
1. Data collection and preprocessing
2. Feature extraction (HTML structure, text content)
3. Model training with SMOTE for class balancing
4. Model evaluation and visualization
5. Saving model artifacts

### Model Artifacts

After training, the following model artifacts will be saved to the `models/` directory:
- `random_forest_model.pkl`: Trained Random Forest model
- `label_encoder.pkl`: Label encoder for categories
- `tfidf_vectorizers.pkl`: TF-IDF vectorizers for text fields


## Project Structure

- `content_classifier.py`: Command-line tool for content type prediction
- `training.ipynb`: Jupyter notebook for model training
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation
- `setup.sh`: Setup script for macOS/Linux users
- `setup.bat`: Setup script for Windows users
- `data/`: Directory for data files (not included in repository)
- `models/`: Directory for saved model artifacts (not included in repository)
