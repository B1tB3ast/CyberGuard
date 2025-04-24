# CyberGuard: Detection of inappropriate content on Social Media indicating Cyberbullying
CyberGuard is an automated system designed to detect cyberbullying on social media using NLP and machine learning techniques, with a focus on multilingual and Hinglish content.

## Directory Structure
```
├── Datasets/                            # Raw and processed datasets
├── image/                               # Visual assets for analysis
├── BERT.ipynb                           # BERT-based model implementation
├── LSTM.py                              # LSTM-based model script
├── data_preprocess.ipynb                # Data cleaning and preprocessing
├── eda.ipynb                            # Exploratory Data Analysis
├── merged_cleaned_dataset_balanced.csv  # Final cleaned and balanced dataset
├── stopwords.txt                        # Custom stopwords list
├── traditional_models.ipynb             # Traditional ML models (SVM, RF, etc.)
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/B1tB3ast/CyberGuard.git
   cd CyberGuard
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks/scripts in the following order:
   - Open and execute `data_preprocess.ipynb` for data cleaning and preprocessing.
   - Open and execute `eda.ipynb` for exploratory data analysis.

4. Choose a model to train and evaluate:
   - For traditional machine learning models, run `traditional_models.ipynb`.
   - For LSTM-based models, execute `LSTM.py`.
   - For BERT-based models, open and run `BERT.ipynb`.
