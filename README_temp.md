the general project

The interactions and the combined effect of 1) the rising temperatures and 2) the degraded Air Quality (AQ),
has put into pressure the living conditions in urban areas. Residents are subjected to hotter conditions
compared to rural areas due to the modulation of the local climate-forming Urban Heat Islands (UHI). Heat
stress from UHI has been found to worsen existing medical conditions, including respiratory disorders,
leading to increased emergency department visits during heat waves (Fuhrmann et. al., 2016). Moreover,
climate change has been shown to amplify the effect of UHI (Founda et. al., 2012, Keppas et al., 2021). In
parallel, anthropogenic emissions from urban mobility (e.g. cars, freight transportation), deteriorate AQ1,
leading to the Urban Pollution Island (UPI) effect. The concept of UPI has been introduced to express the
spatiotemporal variation of pollutant concentrations and to identify hotspots and high pollution events.
(Ulpiani, 2020) conducted a three decade literature review studying the synergy between UHI and UPI and
concluded that urban characteristics play an integral role in the amplification of the effects produced (e.g.
increased heat storage capacity, pollutant trapping). Therefore, aspects such as urban mobility become
predominant factors for understanding environmental phenomena closer to scales where humans operate. For
instance, traffic congestion produces a direct negative effect on urban activities (like pedestrian mobility) by
deteriorating AQ; high concentrations of particulate matter can exacerbate the negative effects of UHI; UHI
creates favourable conditions for pollution episodes; UHI, UPI and traffic congestion contribute to
worsening health issues establishing a vice-versa cycle of interactions between the urban atmospheric
environment and the human activities and living conditions within the urban web. Considering the above, it
can be argued that the effects of UHI and UPI should be considered under a compound perspective, to better
represent the atmospheric environment of smart cities. This is further supported by the conceptual framework
1 It is evident that more than 80% of the population located in the cities are experiencing poor AQ, leading to
400-450 thousand premature deaths (EEA, 2019).
Dimitrios Melas Part B2.1 EnviTwin
[ThA.1] 2│ 16
developed in (Ulpiani, 2020), where it is indicated that many mitigation strategies can simultaneously
address both UHI and UPI.
Designed to promote the well-being and the prosperity of the inhabitants and businesses in urban areas, smart
cities have become the main focal point to better understanding and supporting the processes taking place in
terms of urban metabolism and its corresponding environmental footprint, among others. One of the most
innovative research approaches for the assessment of the effectiveness of smart cities, as well as for its
practical implementation, is the concept of digital twin (DT). A DT is an interactive digital representation of
the urban environment and operates as follows: a functional information feed-back loop runs where 1)
initially the physical world is monitored with advanced sensing technologies, 2) the DT inner models are
updated constantly from established data streams flowing in continuously, 3) employ information and
communication technology to transmit and receive heterogeneous data via the internet of things (IoT)
paradigm in real time, 4) homogenises and summarises the information with analytics, modeling and data
fusion, 5) the knowledge generated is used to improve the quality of life in the real world. On this basis
services to increase awareness and to apply effective decision making can be developed. A DT can provide
solutions from urban planning to land-use management, allowing the simulation of plans and to obviate
adverse scenarios before they happen (Caprari, 2022). Concerning urban AQ, data driven modeling and DT
are promoted as the most efficient route to decision making (Topping, 2021).
In the frame of this proposal, an Environmental digital Twin for smart cities (EnviTwin) will be
implemented, enabling the development of a novel, high resolution, environmental DT that will:
● model the UHI and UPI.
● quantify the synergetic effects of UHI and UPI on urban activities.
● fuse heterogeneous data sources, aiming at increasing the spatiotemporal resolution, utilizing beyond
state-of-the-art (SOTA) methods to reveal areas of increased thermal and respiratory risk at street
level (hotspots).
● update the models with a feed-back loop for investigating real world case studies.
1.2 Proposal objectives and challenges
The EnviTwin is divided into five (5) Scientific and seven (7) Technical Objectives:
Scientific objectives:
SO1: Investigation of non-linear synergies between UHI and UPI with urban mobility.
SO2: Refinement of existing emission inventories and investigation of their impact on numerical modeling.
SO3: Advance the SOTA ML-based data fusion methods for effective spatial disaggregation of
environmental maps at street level.
SO4: Dosimetry modeling and health risk assessment due to particulate air pollution at ultra-high spatial
resolution.
SO5: Assessing the impact of adding environmental aspects to the routing of vehicles and residents in urban
areas.
Technical Objectives:
TO1: Implementation of the modeling tools, update of their emission inventories and unification with real-
time data.
TO2: Processing of heterogeneous datasets collected from IoT networks, satellites (remote sensing),
numerical models and ground based observations in real-time, including crowd-sourced data.
TO3: Implementation of beyond SOTA calibration process for low-cost AQ sensor networks (LCAQSN).
TO4: Implementation of semi-automatic process for estimating urban characteristics (e.g. land use).
TO5: Data fusion and disaggregation of temperature and air quality data at street level in a nowcasting
mode.
TO6: Smart city case study 1: Construction of inhaled dose high-resolution maps for variable-aged
recipients.
Dimitrios Melas Part B2.1 EnviTwin
[ThA.1] 3│ 16
TO7: Smart city case study 2: Adding environmental weight to the cost function of routing engines.
The main challenges in EnviTwin are:
● To produce, collect and harmonize the individual components describing the atmospheric and urban
environment. To tackle this challenge the EnviTwin will exploit and capitalize on existing results
from previous European and national projects related to earth sciences and urban activities, and will
combine them in order to create a DT for the quality of the atmospheric environment in
Thessaloniki.
● To achieve reliable street level information. To address the challenge, AQ data will be utilised, with
different scale and resolution as well as auxiliary spatial information to guide the disaggregation via
ML-based data fusion methods. The results will be evaluated on reference grade instrument
observations from the recently updated national AQ network.
● To quantify the synergetic effects of UHI and UPI. These effects will be studied with the aid of the
numerical modeling system, operating it on switch on/off mode.
● To successfully estimate the inhaled dose caused by airborne particles at multiple locations in a large
scale urban environment. Accomplishment of the challenge will be performed through simulations
that incorporate the available network atmospheric data into an exposure dose model able to produce
deposited dose rates under different scenarios.
● To be able to provide routing/navigation services in urban areas based on the UHI and UPI
conditions of each area. CERTH has developed multiple routing engines for various modes (cars,
public transport, pedestrian) and with various optimization functions (time, cost, exposure to
crowded areas), and will update them with the introduction of environmental costs functions.

specific for this repo

Significant advancements in Low-Cost Air Quality Sensor Network (LCAQSN) calibration have been
realized in recent years (Hofman et. al. 2022, Bagkis et. al. 2022). Graph neural networks (GNN) have
emerged as a modeling technique that has the ability to capture spatiotemporal relationships on non-
Euclidean spaces. Their superior performance has already been established in the field of air quality
forecasting (Ouyang et. al. 2021) as well as in the traffic forecasting field (Ta et. al. 2022). Recently, GNNs
were introduced for the tasks of missing data imputation, virtual sensing, and data fusion (Ferrer-Cid et. al.
202) with remarkable results. However, to the best of our knowledge, GNNs have yet to be tested and
Dimitrios Melas Part B2.1 EnviTwin
[ThA.1] 4│ 16
adjusted for the on-site calibration of LCAQSN, which is one of the gaps this project aims to fill (SO3,
TO3).
In the environmental domain, fusing data from heterogeneous sources like LCAQSN, environmental and
traffic data (Kassandros et al., 2022), has been mostly approached through geostatistical methods, such as
Universal Kriging and Optimal Interpolation (Gressent et. al. 2020). A major issue in the implementation of
EnviTwin is that it should be updated hourly. However, these approaches lack the required computational
optimization to acquire results on sufficient time in ultra-high spatial resolution (~10m2). Moreover, the
introduction of a secondary variable in Cokriging adds significant complexity to the equations of Kriging and
there are limitations in the kernel function in the existing Kriging tools. A novel methodology addressing all
the existing limitations, utilising Gaussian Processes (GPs) will be used in the scope of this project (SO3,
TO5). GPs, mathematically, can be considered a generalisation of Kriging. For a 2-dimensional domain,
exploiting numerous auxiliary variables, accelerating with GPU computations and implementing Deep
Kernel Learning (Gordon et. al. 2015), is expected to effectively fuse the multiple and heterogeneous
datasets.

EnviTwin aims to serve as a mediator between the smart city and the physical world's atmosphere while
holding a near real-time updated high-resolution representation of the latter and support a feed-back loop
between the two. A general flow chart is illustrated in Figure 1. Initially, apart from the static datasets which
will be readily available to the DT system, the datasets will be collected (step 1) and aggregated to hourly
resolution. Data pipelines will be established to automate the data collection to a central server (step 2).
Appropriate preprocessing (e.g. missing data handling, noise removal, feature engineering, downscaling) will
be performed for each dataset (step 3). Furthermore, the data and software infrastructure will handle the
databases and access points of the datasets (step 4). Two machine learning tasks will follow, namely 1) on-
site calibration of the LCAQSN via the expressive spatiotemporal graph neural network models (step 5) and
2) urban characteristics, such as LU/LC and local climate zones (LCZ), estimation from remote sensing data
employing SOTA tree-based ensemble algorithms (step 6). One of the two core components of the EnviTwin
framework is the numerical modeling (both meteorological for UHI and chemical transport for UPI) that will
provide the first representation layer of the atmosphere at regional resolution (step 7). The already
operational numerical models will be adapted to include detailed traffic emissions, LU/LC and LCZ. For all
estimated datasets, validation will take place to ensure data quality (step 8). The other core component is the
data fusion (step 9) that follows, which will spatially disaggregate the UHI and UPI data to urban scale based
on the aforementioned datasets spatiotemporal and covariate correlations. EnviTwin’s new representation
layer propagation (step 10) will allow for the pilot implementation of a green routing and dose response
maps will be calculated with the UPI data thus, quantifying the influence of the physical environment
conditions to the health and mobility of residents. Actionable insights from the two pilots will be provided
back to the physical world therefore, closing the information loop.

This section provides the necessary information about the datasets that will be utilized.
Table 1. Description of the available relevant datasets and their respective source. The datasets that are
provided by each partner are denoted with the abbreviation of the partner. Environmental informatics
research group (EIRG), Atmospheric monitoring and modeling services (ATMOS), and Hellenic institute of
transport (HIT).
Dataset Resolution Project / Provider Status / Acquisition
Urban Pollution Island
Low-cost AQ sensor network Fixed, =<1h KASTOM / EIRG and ATMOS Operational / Sensed
Citizen science AQ sensors Fixed, =<1h Purple air / open source Operational / Sensed
Reference grade AQ monitoring
instruments Fixed, 1h National AQ network and
Municipality AQ network Operational / Sensed
Physical modeling AQ estimations 2km, 1h KASTOM / ATMOS Operational / Modeled (CAMx)
Emissions inventory 2 km, 1h KASTOM / ATMOS Operational / Modeled (NEMO)
Urban Heat Island
Ground based meteorology Fixed, 1h Northmeteo and Darksky / open
source Operational / Sensed
Temperature 250m, 1h LIFE-ASTI / ATMOS Operational / Modeled (WRF)
Urban mobility
Travel time (Bluetooth sensors) Fixed, 15m - / HIT Operational / Sensed
Average speed (Floating car data) Fixed, 15m - / HIT Operational / Sensed
Traffic Congestion Level Fixed, 15m - / HIT Operational / Estimated
Urban characteristics
Remote sensing 10-60m, 5d Sentinel 1 & 2 / Copernicus Operational / Sensed
Land use/land cover 10-60m, Monthly - / EIRG Operational / Modeled (ML)
Local climate zones 50m, Static - / EIRG Existing / Modeled (ML)
Road network topology Segments, Static OpenStreetMap / open source Existing / Compiled
2.1.3 Machine learning and data fusion methods
Dimitrios Melas Part B2.1 EnviTwin
[ThA.1] 7│ 16
The connecting link between the physical, through the networks and the satellite data, and the digital world
will be a machine learning (ML) atmospheric environment data fusion. It will comprise of a) innovative
sensor network calibration and computational improvement (SO3, TO3), b) satellite-based, ML powered fine
scale LU/LC estimations and building height data provision (SO3, TO4) and c) spatial modeling for the
fusion of AQ estimations, that will be able to continuously be improved by the AQ sensor readings,
projecting them to finer spatial scales (SO3, TO5). Graphs are a natural framework to represent LCAQSN
and GNNs fit exceptionally well the on-site calibration requirements (non-Euclidean, non-linear, distributed,
sparse) of such a network. On this basis, aiming to improve the trustworthiness and data quality of the
network, novel GNN based calibration techniques informed by urban mobility and heat stress will be the
subject of investigation. The calibrated measurements of the LCAQSN will be evaluated according to the
European directive 2008/50/EC2, the recently standard ―Air quality - Performance evaluation of air quality
sensor systems - Part 1: Gaseous pollutants in ambient air‖3, and the, soon to be released (2023), part 2 of the
same standard for particulate matter. SOTA ensemble-based algorithms (e.g. random forests, xgboost,
stacking) will be employed for the estimation of spatial characteristics of the environment from satellite
imagery. Specifically, on the basis of an already developed semi-automatic procedure (Katsalis et. al. 2022),
LU/LC raster maps will be reconstructed. External components (e.g. road network topology, LU/LC raster
maps) are expected to provide valuable high resolution spatial information to associate the ground based AQ
measurements with the respective urban characteristics. To approach street level resolution (~10m2), for both
UPI and UHI, the numerical models (WRF, CAMx) estimations will be disaggregated on the basis of a data
fusion methodology. The overarching goal is to combine high and low resolution maps with fixed
measurements, dynamic and static characteristics of the city web. Data fusion will incorporate ML
algorithms and deep learning techniques for their versatility and ability to integrate multiple sources of
relevant information. The GPs models using kernels parameterized by neural networks will be trained on the
available measurement networks (AQ and Temperature) and estimate the predicted surfaces, using auxiliary
variables (traffic, numerical modeling, LU/LC etc). The algorithmic performance will be evaluated with
spatial cross validation on all the available reference stations, which is the appropriate method to establish
reliable metrics in a spatial domain.