# 🔬 Laser Vias Metrology Automation

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![SciPy](https://img.shields.io/badge/SciPy-Scientific-red?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📋 Project Context
**Collaboration with AT&S (Austria) | Computational Data Analysis in Materials Science**

This project automates the quality control analysis of **Laser Vias** used in Integrated Circuit (IC) Substrates. In high-precision manufacturing, validating the geometry of thousands of micro-drilled vias is critical.

This script processes raw sensor data to determine key metrology metrics such as **Taper Angle**, **Roundness**, and **Diameter Distributions**, helping engineers detect deviations in the laser drilling process.

## ⚙️ Key Features
* **Data Ingestion & Cleaning:** Automates the reading of raw CSV process files (`data/`), using **NumPy Masking** to segregate Top vs. Bottom diameter measurements based on dimensional thresholds.
* **Statistical Analysis:** Implements **Gaussian Fitting** (`scipy.optimize.curve_fit`) to model the diameter distribution of the micro-vias.
* **Geometric Calculation:** Algorithms to compute industrial metrics:
    * *Taper Angle* (Side wall inclination in degrees).
    * *Roundness %* (Geometric integrity/circularity).
* **Automated Reporting:** Generates individual plots per process (`outputs/figures/`) and a consolidated textual report (`outputs/overall_report/`) ready for auditing.

## 📂 Repository Structure
The project follows a modular structure where inputs (`data`), logic (`src`), and results (`outputs`) are clearly separated:

```text
laser-vias-metrology-automation/
│
├── data/                       # 📥 INPUT: Raw .csv measurement files
│
├── outputs/                    # 📤 OUTPUT: Generated analysis artifacts
│   ├── figures/                # Visuals (Histograms & Fits) organized by process
│   │   ├── process_1/
│   │   ├── ...
│   │   └── process_8/
│   └── overall_report/         # Final consolidated metrics report (.txt)
│
├── src/                        # 💻 LOGIC: Source code
│   └── laser-vias-metrology-automation.py
│
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

## 📊 Visual Results (outputs/figures)
The script generates histograms comparing raw sensor data against the calculated Gaussian model to identify quality trends. The figures per process are stored within outputs/figures directory.

## 💻 Highlighted Code 
**SnippetGaussian Fit Implementation:**The core of the analysis uses SciPy to fit the normal distribution curve to the noisy sensor data, optimizing parameters ($A$, $\mu$, $\sigma$).

```Python
# Function to define the Gaussian model
def gaussian(x, A, mu, sigma):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Fitting the data using scipy.optimize
# p0 provides initial guesses for Amplitude, Mean, and Std Dev
top_popt, _ = curve_fit(gaussian, bin_centers, top_counts, 
                        p0=[np.max(top_counts), np.mean(top_data), np.std(top_data)])
```
## 🚀 How to Run

**Clone the repository:**

```bash
git clone [https://github.com/tu-usuario/laser-vias-metrology-automation.git](https://github.com/Prado500/laser-vias-metrology-automation.git)
cd laser-vias-metrology-automation
```

**Install dependencies:** It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Run the analysis script:** 
**Important:** Execute the script from the root directory so it can correctly locate the data/ folder.

```bash
python src/laser-vias-metrology-automation.py
```
**Check the results:** As the script is executed, all figures will be displayed on screen, and a .txt file named ´Laser_Vias_Analysis_Results.txt´ will be generated in the root folder. You can access the outputs/ directory to view the generated figures and the Laser_Vias_Analysis_Results.txt report without running the script. 

## 🛠️ Tech Stack

**Language:** Python 3

**Libraries:** 

NumPy & Pandas: Data manipulation, boolean masking, and tabular reporting.

SciPy: Statistical modeling and curve fitting algorithms.

Matplotlib: Data visualization (Histograms and subplot generation).

## 👨‍💻 About the Author
This project was developed during my research internship at **Montanuniversität Leoben (Austria)** 🇦🇹 as part of the MULgrain Excellence Scholarship.

* **Connect with me:** [LinkedIn Profile](https://www.linkedin.com/in/david-alejandro-de-los-reyes-ostos-0b808521a/) (See full portfolio & certifications)
  
* 🚀 **See my Full Stack Work:** [Plasticket App](https://plasticket-app.com) (Award-winning Logistics Platform Demo)
  
**Developed by David Alejandro De Los Reyes Ostos as part of the Computational Data Analysis in Materials Science curriculum (Montanuniversität Leoben).**
