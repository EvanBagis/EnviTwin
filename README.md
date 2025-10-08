# EnviTwin: Environmental Digital Twin for Smart Cities

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com/build-status)
[![Contributors](https://img.shields.io/github/contributors/EvanBagis/EnviTwin)](https://github.com/EvanBagis/EnviTwin/graphs/contributors)

## 🚀 Overview

The EnviTwin project addresses the critical environmental challenges faced by urban areas, specifically the combined effects of rising temperatures (Urban Heat Islands - UHI) and degraded Air Quality (AQ) leading to Urban Pollution Islands (UPI). These phenomena significantly impact living conditions and human health, creating a complex interplay with urban activities like mobility.

EnviTwin proposes an innovative Environmental Digital Twin for smart cities, designed to provide a novel, high-resolution representation of the urban atmospheric environment. It aims to:

*   Model UHI and UPI effects.
*   Quantify the synergistic impacts of UHI and UPI on urban activities.
*   Fuse heterogeneous data sources using advanced methods to reveal areas of increased thermal and respiratory risk at street level.
*   Continuously update models through a feedback loop for real-world case studies.

## 🎯 Objectives and Challenges

### Scientific Objectives

*   **SO1:** Investigate non-linear synergies between UHI, UPI, and urban mobility.
*   **SO2:** Refine existing emission inventories and study their impact on numerical modeling.
*   **SO3:** Advance State-Of-The-Art (SOTA) ML-based data fusion methods for effective spatial disaggregation of environmental maps at street level.
*   **SO4:** Perform dosimetry modeling and health risk assessment due to particulate air pollution at ultra-high spatial resolution.
*   **SO5:** Assess the impact of integrating environmental aspects into vehicle and resident routing in urban areas.

### Technical Objectives

*   **TO1:** Implement modeling tools, update emission inventories, and unify with real-time data.
*   **TO2:** Process heterogeneous datasets from IoT networks, satellites, numerical models, and ground observations in real-time.
*   **TO3:** Implement beyond SOTA calibration processes for Low-Cost Air Quality Sensor Networks (LCAQSN).
*   **TO4:** Implement semi-automatic procedures for estimating urban characteristics (e.g., land use).
*   **TO5:** Perform data fusion and disaggregation of temperature and air quality data at street level in a nowcasting mode.
*   **TO6:** Smart city case study 1: Construct high-resolution inhaled dose maps for variable-aged recipients.
*   **TO7:** Smart city case study 2: Integrate environmental weight into routing engine cost functions.

### Main Challenges

*   Producing, collecting, and harmonizing individual components describing the atmospheric and urban environment.
*   Achieving reliable street-level information.
*   Quantifying the synergistic effects of UHI and UPI.
*   Successfully estimating the inhaled dose caused by airborne particles at multiple locations.
*   Providing routing/navigation services based on UHI and UPI conditions.

## 💡 Key Advancements (Specific to this Repository)

This repository focuses on significant advancements in:

*   **LCAQSN Calibration:** Utilizing Graph Neural Networks (GNNs) to capture spatiotemporal relationships for on-site calibration of LCAQSN, addressing a gap in current methodologies (SO3, TO3). GNNs are particularly suited for the non-Euclidean, non-linear, distributed, and sparse nature of sensor networks.
*   **Data Fusion:** Employing novel methodologies based on Gaussian Processes (GPs), a generalization of Kriging, for fusing heterogeneous environmental and traffic data. This approach, accelerated with GPU computations and Deep Kernel Learning, aims to overcome limitations of traditional geostatistical methods in achieving ultra-high spatial resolution and computational efficiency (SO3, TO5).

## 🔄 EnviTwin Framework Flow

The EnviTwin framework operates through a 10-step information feedback loop:

1.  **Data Collection:** Datasets are collected and aggregated to hourly resolution.
2.  **Data Pipelines:** Automated data collection to a central server.
3.  **Preprocessing:** Missing data handling, noise removal, feature engineering, and downscaling.
4.  **Infrastructure:** Management of databases and access points for datasets.
5.  **ML Task 1 (Calibration):** On-site calibration of LCAQSN via expressive spatiotemporal GNN models.
6.  **ML Task 2 (Estimation):** Urban characteristics (LU/LC, LCZ) estimation from remote sensing data using SOTA tree-based ensemble algorithms.
7.  **Numerical Modeling:** Meteorological (UHI) and chemical transport (UPI) models provide the first representation layer at regional resolution.
8.  **Validation:** Data quality assurance for all estimated datasets.
9.  **Data Fusion:** Spatial disaggregation of UHI and UPI data to urban scale based on spatiotemporal and covariate correlations.
10. **Propagation & Pilots:** New representation layer propagation enables pilot implementations like green routing and dose response maps, closing the information loop.

## 📊 Datasets

The project utilizes a variety of heterogeneous datasets, including:

*   **Urban Pollution Island:** Low-cost AQ sensor networks, citizen science AQ sensors, reference grade AQ monitoring instruments, physical modeling AQ estimations (CAMx), emissions inventory (NEMO).
*   **Urban Heat Island:** Ground-based meteorology, temperature data (WRF).
*   **Urban Mobility:** Travel time (Bluetooth sensors), average speed (Floating car data), traffic congestion level.
*   **Urban Characteristics:** Remote sensing (Sentinel 1 & 2), Land Use/Land Cover, Local Climate Zones, Road network topology (OpenStreetMap).

## 🛠️ Machine Learning and Data Fusion Methods

The core of EnviTwin's connection between the physical and digital worlds is an ML-driven atmospheric environment data fusion, comprising:

*   **Innovative Sensor Network Calibration:** GNN-based calibration techniques, informed by urban mobility and heat stress, to improve the trustworthiness and data quality of LCAQSN.
*   **Satellite-based LU/LC Estimations:** SOTA ensemble-based algorithms (e.g., random forests, xgboost) for reconstructing LU/LC raster maps from satellite imagery.
*   **Spatial Modeling for AQ Fusion:** Gaussian Processes (GPs) models, using kernels parameterized by neural networks, to combine high and low-resolution maps with fixed measurements, dynamic, and static city characteristics. This aims to disaggregate numerical model estimations (WRF, CAMx) to street-level resolution (~10m²).

## ⚙️ Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EvanBagis/EnviTwin.git
    cd EnviTwin
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r Operational_imputation_and_calibration/requirements.txt
    # Add any other requirements files if necessary
    ```
    *(Note: You may need to install additional system-level dependencies depending on your environment and the specific libraries used.)*

## 🚀 Usage

This section will provide instructions on how to run the different modules and scripts within the EnviTwin project.

### Operational Imputation and Calibration

*   **To run the operational imputation script:**
    ```bash
    python Operational_imputation_and_calibration/operational_imputation.py [arguments]
    ```
*   **To run the historical calibration script:**
    ```bash
    python Operational_imputation_and_calibration/historical_calibration.py [arguments]
    ```
    *(Refer to the individual script files for specific command-line arguments and configuration details.)*

### Operational LULC

*   **To run the LULC training/prediction script:**
    ```bash
    python Operational_LULC/scripts/Unet_train_predict.py [arguments]
    ```
*   **To create configuration files:**
    ```bash
    python Operational_LULC/scripts/create_config.py [arguments]
    ```
    *(Refer to the individual script files for specific command-line arguments and configuration details.)*

## 📂 Project Structure

```
.
├── Operational_imputation_and_calibration/  # Code for imputation and calibration models
│   ├── calibration_models/                  # Stored calibration models (ignored by .gitignore)
│   ├── predictions/                         # Prediction outputs (ignored by .gitignore)
│   ├── generate_graphs_historical.py
│   ├── historical_calibration.py
│   ├── operational_calibration.py
│   ├── operational_imputation.py
│   ├── requirements.txt
│   └── tst.ipynb                            # Test notebook (ignored by .gitignore)
├── Operational_LULC/                        # Code for Land Use/Land Cover estimation
│   ├── config.json
│   ├── labels/                              # Label files
│   ├── models/                              # Stored LULC models (ignored by .gitignore)
│   ├── scripts/
│   │   ├── Unet_train_predict.py
│   │   ├── create_config.py
│   │   └── download_polished.py
│   └── .gitignore                           # Specific ignores for LULC module
├── .gitignore                               # Global ignore rules
└── README.md                                # Project overview (this file)
# ... other directories and files (e.g., Fusion, Operational_data_fusion, EnviTwin_venv - ignored)
```

## 🤝 Contributing

We welcome contributions to the EnviTwin project! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) (if available) for guidelines on how to submit issues, propose features, and make pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any inquiries or support, please contact [Your Name/Email/Organization].
