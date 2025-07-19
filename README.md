# 🎬 Movie Box Office Analysis

A comprehensive data science project analyzing movie box office performance using machine learning techniques to predict revenue and identify key success factors.

**Project By:** Faizal Ardian Putra

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Key Features](#key-features)
4. [Analysis Results](#analysis-results)
5. [Machine Learning Models](#machine-learning-models)
6. [Key Insights](#key-insights)
7. [Technologies Used](#technologies-used)
8. [Getting Started](#getting-started)
9. [Project Structure](#project-structure)
10. [Business Recommendations](#business-recommendations)

## 🎯 Project Overview

This project analyzes movie data to understand factors that influence box office success, identify trends in the movie industry, and build predictive models for movie revenue. The analysis includes comprehensive exploratory data analysis (EDA), feature engineering, and machine learning model development.

### Goals
- Understand factors that influence box office success
- Identify trends in the movie industry
- Build a predictive model for movie revenue
- Create insightful visualizations
- Provide actionable business recommendations

## 📊 Dataset

The project uses a synthetic dataset of **1,500 movies** spanning from 2000-2024 with the following features:

- **Title**: Movie identifier
- **Genre**: Action, Comedy, Drama, Horror, Romance, Sci-fi, Thriller
- **Year**: Release year (2000-2024)
- **Budget**: Production budget in millions USD
- **Runtime**: Movie duration in minutes
- **IMDb Rating**: Rating score (1-10)
- **Director Experience**: Years of director experience
- **Star Power**: Celebrity rating (1-10)
- **Studio Size**: Major, Mid-tier, Independent
- **Revenue**: Box office revenue in millions USD

## 🔍 Key Features

- **Data Cleaning & Preprocessing**: Comprehensive data validation and feature engineering
- **Exploratory Data Analysis**: Statistical analysis and visualization of movie trends
- **Machine Learning Models**: Linear Regression and Random Forest for revenue prediction
- **Feature Importance Analysis**: Identification of key success factors
- **Business Intelligence**: Actionable insights for movie industry stakeholders

## 📈 Analysis Results

### Model Performance
- **Best Model**: Random Forest Regressor
- **Accuracy**: 64.6% (R² Score: 0.646)
- **RMSE**: $17.14 million

### Key Statistics
- **Total Movies Analyzed**: 1,500
- **Average Revenue**: $42.3M
- **Average ROI**: 183.4%
- **Time Period**: 2000-2024

## 🤖 Machine Learning Models

### Models Implemented
1. **Linear Regression**
   - R² Score: 0.559
   - RMSE: $19.15M
   - Accuracy: 55.9%

2. **Random Forest Regressor** ⭐ (Best Performing)
   - R² Score: 0.646
   - RMSE: $17.14M
   - Accuracy: 64.6%

### Feature Importance (Random Forest)
1. **Star Power**: 47.1% - Most influential factor
2. **IMDb Rating**: 15.6% - Critical for success
3. **Studio Size**: 13.4% - Distribution impact
4. **Genre**: 7.9% - Market preferences
5. **Runtime**: 7.0% - Audience engagement
6. **Director Experience**: 4.7% - Creative leadership
7. **Year**: 4.3% - Market trends
8. **Budget**: 0.0% - Surprisingly minimal direct impact

## 💡 Key Insights

### Revenue Performance by Genre
1. **Action**: $55.8M average (Highest grossing)
2. **Sci-fi**: $51.2M average
3. **Thriller**: $49.1M average
4. **Horror**: $46.8M average
5. **Comedy**: $44.2M average
6. **Romance**: $38.9M average
7. **Drama**: $33.8M average (Lowest grossing)

### Studio Performance
- **Major Studios**: Highest revenue potential and distribution reach
- **Mid-tier Studios**: Balanced performance and ROI
- **Independent Studios**: Lower revenue but potentially higher creative freedom

### Success Factors
- **Star Power** is the most critical factor (47% importance)
- **IMDb Rating** significantly impacts box office performance
- **Genre choice** affects revenue potential significantly
- **Studio partnerships** provide distribution advantages

## 🛠 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms
- **Jupyter Notebook** - Interactive development environment

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Analysis
1. Clone this repository
2. Open `movie_analysis.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells to reproduce the analysis
4. Explore the visualizations and insights

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/movie-analysis-google-collab/blob/main/movie_analysis.ipynb)

## 📁 Project Structure

```
movie-analysis-google-collab/
│
├── movie_analysis.ipynb    # Main analysis notebook
├── README.md              # Project documentation
└── data/                  # Data files (if external data used)
```

## 💼 Business Recommendations

Based on our comprehensive analysis, here are key recommendations for movie industry stakeholders:

### For Producers & Studios
1. **Focus on Action/Sci-Fi genres** for maximum revenue potential
2. **Invest in star power** - Most critical success factor (47% importance)
3. **Target IMDb rating of 7.0+** for better box office performance
4. **Partner with major studios** for distribution advantages
5. **Optimize runtime to 90-120 minutes** for best ROI balance

### For Investors
1. **Action + Major Studio combinations** show highest ROI (383.6%)
2. **Probability of top 25% revenue**: Action + Major Studio (60.2%)
3. **Budget allocation**: Higher budgets correlate with higher revenues
4. **Risk assessment**: Independent studios offer lower but more predictable returns

### Success Probability Analysis
**Top Performing Combinations** (Probability of achieving top 25% revenue):
- Action + Major Studio: 60.2%
- Thriller + Major Studio: 43.1%
- Horror + Major Studio: 41.5%
- Action + Mid-tier Studio: 41.4%
- Thriller + Mid-tier Studio: 39.3%

## 📊 Visualizations

The project includes comprehensive visualizations:
- Revenue trends over time
- Genre performance comparison
- Budget vs Revenue correlation analysis
- Feature importance charts
- ROI distribution analysis
- Studio performance comparison
- Rating vs Revenue relationships

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

**Faizal Ardian Putra**
- Email: [your-email@example.com]
- LinkedIn: [your-linkedin-profile]
- GitHub: [your-github-username]

---

*This project demonstrates end-to-end data science workflow including data analysis, machine learning, and business intelligence for the entertainment industry.*