# 🌬️ Project FreshAir: Air Pollution and COVID-19

An analytical project exploring the relationship between air quality (specifically PM2.5 pollution) and the spread of COVID-19 in France. The project includes API data collection, visualization, and statistical correlation analysis between pollution levels and health outcomes.

---

## 📌 Project Goal

To investigate whether air pollution levels — particularly PM2.5 — had an impact on the rise of COVID-19 cases across different French regions between 2018 and 2023.

---

## 📊 Key Research Topics

- Exploring the link between air quality and COVID-19 outcomes
- Air pollution trends before, during, and after the pandemic
- Monthly COVID-19 case statistics across France
- Comparative analysis of two French regions: **Auvergne-Rhône-Alpes** and **Île-de-France**
- Visualizing and interpreting data correlations

---

## 🔧 Technologies Used

- **Language:** Python  
- **Libraries:** pandas, matplotlib, seaborn, requests  
- **Data Sources:**
  - [OpenAQ API](https://docs.openaq.org/) — historical air pollution data
  - [disease.sh API](https://disease.sh/) — COVID-19 case data
  - Dataset from [data.gouv.fr](https://www.data.gouv.fr/fr/)
  - Final results presented in a PDF slide deck

---

## 📁 Project Structure

📦 Project_FreshAir
'''├── freshair_analysis.py # Main data analysis script
├── Presentation_FreshAir.pdf # Final presentation slides
├── Sources/ # Reference materials and notes
└── README.md # Project description (this file)'''

---

## 📈 Key Findings

- **Auvergne-Rhône-Alpes**: moderate positive correlation (~0.3) between PM2.5 levels and monthly COVID-19 case counts — suggesting a potential link between pollution and hospitalization rates.
- **Île-de-France**: near-zero correlation — air pollution appeared to have little or no effect on COVID-19 trends in this region.
- PM2.5 levels dipped slightly during the pandemic lockdowns, but not significantly beyond regular seasonal variation.
- A spike in PM2.5 occurred across all regions in late 2022 to early 2023 — possibly due to cold weather, heating, and post-COVID rebound in human activity.

---

## 🚧 Challenges Faced

- Most pollution APIs provided limited historical data (often only from 2022 onward)
- Manual configuration of geographic bounding boxes was required to collect region-specific data
- Lack of detailed API access to respiratory disease data beyond COVID-19
- Merging and cleaning data from multiple sources required extensive preprocessing

---


👥 Authors
Tamara Melioranskaia

Max Pisolkar

🧠 Conclusion
The link between air pollution and COVID-19 varies by region. While no universal pattern was found, moderate positive correlations in certain areas suggest that air quality may influence public health outcomes during pandemics. Continued monitoring and region-specific analysis is essential.

📚 Data Sources
OpenAQ API

disease.sh API

Data.gouv.fr


---

