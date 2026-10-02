<<<<<<< HEAD
\# 🌾 AnnadataAI — Crop Price Intelligence for Andhra Pradesh Farmers



A Machine Learning-powered crop price prediction and decision-support system

built for farmers in Andhra Pradesh, India.



\## 🎯 Project Overview

\- \*\*Model:\*\* Random Forest Regressor

\- \*\*Accuracy:\*\* R² = 0.893 (89.3%)

\- \*\*MAE:\*\* ₹281/quintal

\- \*\*Dataset:\*\* Agmarknet / data.gov.in (2,97,181 records, 2023–2025)

\- \*\*Focus Region:\*\* Andhra Pradesh (30 mandis, 5 crops)



\## 🚀 Features

\- Potato price prediction using lag \& rolling features

\- MSP Violation Detection (Agricultural Economics integration)

\- District-wise price disparity analysis (Geographic integration)

\- Interactive Streamlit web application



\## 📊 Key Findings

\- Best AP mandi: \*\*Adilabad Rythu Bazar\*\* (avg ₹4,000+/quintal)

\- Best selling month: \*\*September\*\* (peak prices every year)

\- \~18% of AP transactions fell below Government MSP



\## 🛠️ Tech Stack

\- Python 3.11, pandas, numpy, scikit-learn

\- matplotlib, seaborn, Streamlit



\## ▶️ Run the App

```bash

pip install streamlit pandas numpy scikit-learn matplotlib seaborn

python -m streamlit run app.py

```



\## 📂 Dataset

Download from Kaggle and place in project folder:

`Agriculture\_price\_dataset.csv`



\## 🎓 Project Info

\- \*\*College:\*\* BVC Engineering College, Odalarevu

\- \*\*Program:\*\* SOIP 2026 — ML Using Python

\- \*\*Guide:\*\* G. Ganga Bhavani, Associate Professor

=======
# 🌾 AnnadataAI — Crop Price Intelligence for Andhra Pradesh Farmers

A Machine Learning-powered crop price prediction and decision-support system built for farmers in Andhra Pradesh, India.

## 🎯 Project Overview
| | |
|---|---|
| **Model** | Random Forest Regressor |
| **Accuracy** | R² = 0.893 (89.3%) |
| **MAE** | ₹281/quintal |
| **Dataset** | Agmarknet / data.gov.in (2,97,181 records, 2023–2025) |
| **Focus Region** | Andhra Pradesh — 30 mandis, 5 crops |

## 🚀 Features
- Potato price prediction using lag & rolling features
- MSP Violation Detection — ML + Agricultural Economics
- District-wise price disparity analysis — ML + Regional Geography
- Interactive Streamlit web application with 4 tabs

## 📊 Key Findings
- **Best AP mandi:** Adilabad Rythu Bazar (avg ₹4,000+/quintal)
- **Best selling month:** September (peak prices every year)
- **2024 price spike:** ₹2,000 → ₹2,870 (+43%)
- **~18%** of AP transactions fell below Government MSP

## 🗂️ Repository Contents
| File | Description |
|---|---|
| `AnnadataAI_Project.ipynb` | Complete ML pipeline notebook |
| `app.py` | Streamlit web application |
| `plot1_price_distribution.png` | AP crop price distribution |
| `plot2_price_trend.png` | Monthly price trend 2023–2025 |
| `plot3_mandi_comparison.png` | Top 10 AP mandis by price |
| `plot4_seasonality.png` | Seasonality heatmap |
| `plot6_actual_vs_predicted.png` | Model prediction results |
| `plot7_feature_importance.png` | Feature importance chart |
| `plot8_msp_analysis.png` | MSP violation analysis |
| `plot9_district_disparity.png` | District price gap analysis |

## 🛠️ Tech Stack
Python 3.11 | pandas | numpy | scikit-learn
matplotlib | seaborn | Streamlit

## ▶️ Run the App
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
python -m streamlit run app.py
```

## 📂 Dataset
Download `Agriculture_price_dataset.csv` from Kaggle:
[Indian Agricultural Mandi Prices Dataset](https://www.kaggle.com)

Place it in the same folder as `app.py` before running.

## 🎓 Project Details
| | |
|---|---|
| **College** | BVC Engineering College, Odalarevu |
| **University** | JNTU-Kakinada |
| **Program** | SOIP 2026 — Machine Learning Using Python |
| **Student** | Krishna Balaji Kothapalli (23221A0565) |

## 📌 Domain Integration
- **ML + Agricultural Economics** → MSP violation detection redirects farmers to government procurement
- **ML + Regional Geography** → District price gap identifies underpaid regions in AP
>>>>>>> e88ae77645cb139c7ca2bea2539e056d487039f8
