# 🎥 IMDb Review Classifier — Turning Text into Insight

> Can a machine understand a movie review?  
> With the right preprocessing, vectorization, and models — **yes, it can.**

---

## 🔍 What’s This Project About?

This project takes 50,000 raw IMDb movie reviews with sentiment and builds a complete **sentiment analysis pipeline** using Python, NLP, and Machine Learning. 

Built from scratch in Python, this project showcases how a well-crafted pipeline — from text cleaning to vectorization to model selection — can produce impressive accuracy even with simple models.

---

## 💡 Key Highlights

✅ **Text Cleaning & Preprocessing**  
- Removed HTML tags, punctuation, special characters  
- Applied **lemmatization** using NLTK  
- Removed stopwords for cleaner signal  

✅ **Efficient Data Processing**  
- Used **Swifter** to accelerate `.apply()` functions  
- Leveraged **regular expressions** for robust text pattern handling  

✅ **Feature Engineering**  
- Converted text into vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**  
- Tuned max features and n-gram ranges  

✅ **Modeling & Evaluation**  
- Built and evaluated two strong baseline models:  
  - **Logistic Regression**  
  - **Multinomial Naive Bayes**  
- Compared using accuracy, precision, recall, and F1-score  

---

## 🧠 Technologies Used

- 🐍 **Python 3**  
- 🛠 **Pandas**, **NumPy**  
- 📚 **NLTK** for NLP tasks  
- 📐 **Scikit-learn** for modeling and evaluation  
- ⚡ **Swifter** for faster DataFrame operations  
- 🧹 **Regex**, `string`, and `re` modules for text cleaning

---

## 🎯 Project Goal

To train supervised ML models on labeled IMDb movie reviews to predict sentiment, using end-to-end NLP techniques — from raw text to vectorized features to final classification — and demonstrate that even baseline models can achieve strong performance when the data pipeline is solid.

---

## 📊 Results

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | 86.1%    |
| Multinomial Naive Bayes | 83.3%    |

🔍 **Observation**: Logistic Regression outperformed Naive Bayes, likely due to its ability to weigh features more flexibly in high-dimensional TF-IDF space.

---

📘 Learnings & Takeaways
✔ Text data is messy, but structured preprocessing can create clarity
✔ Simple models can go a long way with the right features
✔ Building your own pipeline teaches far more than using prebuilt tools
✔ Even with 50,000 reviews, speed matters — swifter and vectorizers save time

🙋‍♀️ About Me
I'm a budding data scientist with a passion for NLP, modeling, and turning raw data into real-world impact.
Connect with me to talk data, code, or cinema 🎬

🔗 GitHub: mubashera124

💼 LinkedIn: https://www.linkedin.com/in/mubashera-siddiqui-59489823a/



