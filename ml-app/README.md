# 🍽️ Restaurant Rating Predictor (Machine Learning Project)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)

### 🇲🇲 မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်
ဤ Project သည် **Kaggle** မှ စားသောက်ဆိုင်ပေါင်း ၉,၅၀၀ ကျော်၏ Data များကို အသုံးပြု၍ ဆိုင်တစ်ဆိုင်၏ ဝန်ဆောင်မှု၊ စျေးနှုန်းနှင့် လူကြိုက်များမှုအပေါ် မူတည်ပြီး ရရှိနိုင်မည့် **Rating Score** ကို ခန့်မှန်းပေးသော Machine Learning App တစ်ခု ဖြစ်ပါသည်။ Full-stack Developer တစ်ယောက်အနေဖြင့် ML Model တည်ဆောက်ခြင်းမှသည် Live Deployment အထိ End-to-End လုပ်ဆောင်ထားသော Portfolio Project တစ်ခု ဖြစ်သည်။

---

## 🚀 Project Overview
This application is a **Restaurant Rating Predictor** built as a machine learning portfolio project. It demonstrates an end-to-end ML pipeline from data acquisition to a live interactive web app.

### 🔗 Live Demo
Check out the live app here: [Live App Link](https://83ade8ab-6b9c-4e43-9178-80ef27d864f1-00-2u4ny8v4qq2sd.worf.replit.dev/)

---

## 📊 Dataset Information
* **Source:** Zomato Restaurants Data on Kaggle.
* **Size:** ~9,551 restaurants worldwide.
* **Usable Rows:** ~7,403 (after cleaning and removing unrated entries).

### Features Used:
1. **Average Cost for two:** Local currency based dining cost.
2. **Price range:** Scale of 1 (Budget) to 4 (Luxury).
3. **Votes:** Total number of user reviews/votes received.

---

## 🧠 Machine Learning Pipeline
1. **Data Loading:** CSV read with `latin-1` encoding.
2. **Cleaning:** Handled missing values and removed zero-rated entries.
3. **Feature Engineering:** Numeric coercion and data normalization.
4. **Model:** Used **RandomForestRegressor** (200 trees) for high accuracy.
5. **Evaluation:** Tested using R-squared and RMSE metrics.



---

## 💡 Key Findings
* **Votes (လူကြိုက်များမှု):** ဆိုင်တစ်ဆိုင်ရဲ့ Rating ကို ခန့်မှန်းတဲ့နေရာမှာ ၇၄% ခန့် အရေးအပါဆုံး ဖြစ်ကြောင်း တွေ့ရှိရသည်။
* **Price Range:** စျေးနှုန်းအမျိုးအစားသည် ခန့်မှန်းချက်အပေါ် ၅% ခန့် သက်ရောက်မှုရှိသည်။

---

## 🛠️ Tech Stack
| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.11 |
| **ML Library** | Scikit-learn |
| **Data Handling** | Pandas, Numpy |
| **Web UI** | Streamlit |
| **Visualization** | Plotly |

---

## ⚙️ How to Run Locally
To run this project on your machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project.git](https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project.git)
