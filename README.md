# 🕵️ Brand Monitoring Sentiment Analysis System

A machine learning project for **classifying social media posts** as **positive** or **negative sentiment** to support brand monitoring.

---


### Step 1: Data Collection & Cleaning
- **Dataset:** Sentiment140 (1.6M tweets)  
- **Sample Size:** 100,000 balanced tweets (50K positive + 50K negative)  

**Text Cleaning Pipeline:**
- Remove URLs, mentions, hashtags, emojis  
- Normalize text (lowercasing, lemmatization)  
- Remove stopwords and irrelevant content  
- Handle missing values and duplicates  

**Output:** ~85,000 clean tweets ready for ML  

---

### Step 2: Feature Engineering
- **TF-IDF Vectorization:** Convert text to numerical features  
- **Feature Matrix:** 85K tweets × 10K features  

**Vocabulary Control:**
- Top 10,000 most important words/phrases  
- Minimum frequency: 5 occurrences  
- Maximum frequency: 95% of documents  
- N-grams: Unigrams + bigrams  

**Data Split:**
- 80% training (68K)  
- 20% testing (17K)  

**Feature Analysis:** Importance ranking and visualization tools 


##  License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
