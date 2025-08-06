# Sequential Data Using ML


## Overview

This comprehensive project explores various forms of sequential data and their applications in machine learning. It demonstrates how to work with non-i.i.d. (non-identically and independently distributed) data, breaking the traditional assumption in machine learning that data points are independent of each other.

## Key Concepts

### Background

In traditional machine learning, we assume data is **identically and independently distributed (i.i.d.)**. However, sequential data violates this assumption because:

- Samples depend on past information
- Data points are reliant on one another due to their sequential order
- Examples include weather patterns, financial time series, speech, text, and biological sequences

### Sequential Data Types Covered

1. **Time Series Data**
   - Financial stock quotes from energy companies
   - Seasonal decomposition and analysis
   - Missing data imputation techniques

2. **Audio Data**
   - Speech command recognition
   - Audio signal processing
   - Spectrogram generation and visualization
   - TensorFlow/Keras integration

3. **Text Data**
   - Natural language processing techniques
   - Tokenization and preprocessing
   - Vector representations

4. **Genomic Data**
   - DNA sequence analysis
   - K-mer decomposition
   - Gene family classification using machine learning

## Technologies Used

### Libraries and Frameworks
- **pandas** - Data manipulation and analysis
- **numpy** - Mathematical operations
- **scikit-learn** - Machine learning algorithms and preprocessing
- **seaborn** - Statistical data visualization
- **matplotlib** - Plotting and visualization
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **SciPy** - Scientific computing
- **statsmodels** - Statistical modeling

### Key Techniques Demonstrated
- Time series decomposition
- Audio signal processing
- Spectrogram analysis
- Text preprocessing and vectorization
- DNA sequence k-mer analysis
- Bag-of-Words modeling
- Naive Bayes classification

## 1: Working with Speech Commands

This section demonstrates audio data processing and analysis:

### Key Features:
- **Data Import**: Loading speech command datasets from TensorFlow
- **Audio Processing**: Converting audio files to numerical representations
- **Visualization**: Plotting waveforms and spectrograms
- **Preprocessing**: Normalizing and reshaping audio data
- **Feature Extraction**: Creating spectrograms for machine learning

### Techniques Covered:
```python
# Audio loading and processing
sample_rate, audio_data = wavfile.read(audio_file)

# Spectrogram generation
frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)

# Visualization
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
```

## 2: Gene Family Classification

This section explores bioinformatics and genomic data analysis:

### Dataset:
- Human DNA sequences with gene family classifications
- 4,380 sequences across 7 different gene families
- Tab-separated format with sequence and class information

### Key Processes:

#### 1. Data Loading and Exploration
```python
human_data = pd.read_csv("human_data.txt", sep="\t")
human_data['class'].value_counts().plot.bar()
```

#### 2. K-mer Decomposition
Converting long DNA sequences into overlapping k-mer subsequences:
```python
def kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
```

#### 3. Bag-of-Words Model
Creating feature vectors from biological sequences:
```python
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
```

#### 4. Machine Learning Classification
Using Multinomial Naive Bayes for gene family prediction:
```python
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
```

### Results:
The classification model achieves excellent performance with detailed metrics shown below:

```
              precision    recall  f1-score   support

           0       0.98      0.97      0.98       102
           1       1.00      0.98      0.99       106
           2       1.00      1.00      1.00        78
           3       0.99      0.99      0.99       125
           4       0.99      0.96      0.98       149
           5       1.00      1.00      1.00        51
           6       0.96      0.99      0.98       265

   micro avg       0.98      0.98      0.98       876
   macro avg       0.99      0.99      0.99       876
weighted avg       0.98      0.98      0.98       876
```

**Key Performance Highlights:**
- **Overall accuracy**: 98% across all gene family classes
- **Perfect classification** (1.00 precision and recall) for classes 2 and 5
- **Consistently high performance** with precision scores ranging from 0.96-1.00
- **Balanced results** with F1-scores between 0.98-1.00 for all classes
- **Robust model** with 876 test samples across 7 gene families

## Financial Time Series Analysis

The notebook includes analysis of financial data from energy companies:

### Companies Analyzed:
- Total (TOT)
- Exxon (XOM)  
- Chevron (CVX)
- ConocoPhillips (COP)
- Valero Energy (VLO)

### Techniques:
- Time series visualization
- Correlation analysis between stock prices
- Demonstration of temporal dependencies in financial data

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow keras
pip install nltk scipy statsmodels
```

### NLTK Data Downloads
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
```






