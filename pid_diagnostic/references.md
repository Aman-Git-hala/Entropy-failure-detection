# References

All references used in the PID Architecture Selection Diagnostic framework, ordered by relevance to the paper's contribution.

---

## Core Framework

### Partial Information Decomposition

1. **Williams, P.L. & Beer, R.D.** (2010). *Nonnegative Decomposition of Multivariate Information.* arXiv:1004.2515.
   - Introduced the PID framework: decomposing mutual information into redundancy, unique information, and synergy atoms.
   - Proposed the I_min redundancy measure and the lattice-based decomposition structure.
   - **Our use**: Foundation for the Synergy Ratio diagnostic.

2. **Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N.** (2014). *Quantifying Unique Information.* Entropy, 16(4), 2161–2183. DOI: 10.3390/e16042161
   - Proposed the BROJA (Bertschinger-Rauh-Olbrich-Jost-Ay) unique information measure.
   - More theoretically justified than I_min for quantifying what each source uniquely contributes.
   - **Our use**: Alternative PID estimator via the `dit` library.

3. **Griffith, V. & Koch, C.** (2014). *Quantifying Synergistic Mutual Information.* In Guided Self-Organization: Inception (pp. 159–190). Springer. DOI: 10.1007/978-3-642-53734-9_6
   - Formal treatment of synergy in multi-source information decomposition.
   - **Our use**: Theoretical justification for the Synergy Ratio metric.

### Shannon Information Theory

4. **Shannon, C.E.** (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27(3), 379–423.
   - Foundation of information theory. Binary entropy H(p) = −p log₂(p) − (1−p) log₂(1−p).
   - **Our use**: Core meta-feature — per-group prediction entropy as uncertainty signal.

5. **Kullback, S. & Leibler, R.A.** (1951). *On Information and Sufficiency.* Annals of Mathematical Statistics, 22(1), 79–86.
   - KL divergence D_KL(P || Q) as a measure of distribution divergence.
   - **Our use**: Cross-group disagreement meta-feature in the 12D vector.

---

## Datasets

### AI4I 2020 (Predictive Maintenance)

6. **Matzka, S.** (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* IEEE 3rd Int. Conf. AI for Industries (AI4I), pp. 391–395. DOI: 10.1109/AI4I49448.2020.00023
   - Introduced the AI4I 2020 dataset: 10,000 CNC machine samples, 6 sensor features, binary failure label.
   - Original RF baseline: AUC = 0.954.
   - **Our use**: Primary benchmark (low synergy, single dominant group — Wear).

### C-MAPSS (Turbofan Degradation)

7. **Saxena, A., Goebel, K., Simon, D., & Eklund, N.** (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM Conference.
   - NASA's Commercial Modular Aero-Propulsion System Simulation dataset.
   - 100 engines, 21 sensor channels, run-to-failure trajectories.
   - **Our use**: Second data point (low synergy, high redundancy — all sensors track single HPC degradation).

8. **Ramasso, E. & Saxena, A.** (2014). *Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets.* International Journal of PHM, 5(2).
   - Standard benchmarks and evaluation methodology for C-MAPSS.

### Server Machine Dataset (SMD)

9. **Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D.** (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network.* KDD 2019.
   - Server Machine Dataset: 28 machines, 38 features, labeled anomalies.
   - **Our use**: Real distributed systems dataset with potential for moderate synergy.

---

## Methods

### Stochastic Gradient Descent

10. **Bottou, L.** (2010). *Large-Scale Machine Learning with Stochastic Gradient Descent.* COMPSTAT 2010, pp. 177–186. DOI: 10.1007/978-3-7908-2604-3_16
    - Foundation for SGD-based logistic regression used in inner and outer models.

### SMOTE (Imbalanced Learning)

11. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
    - Synthetic oversampling for class imbalance. Applied inside CV folds only.

### Probability Calibration

12. **Zadrozny, B. & Elkan, C.** (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* KDD 2002.
    - Isotonic regression for probability calibration.
    - **Our use**: CalibratedClassifierCV with isotonic method.

### Evaluation

13. **Youden, W.J.** (1950). *Index for Rating Diagnostic Tests.* Cancer, 3(1), 32–35.
    - Youden's J statistic for optimal threshold selection: J = TPR - FPR.

14. **Matthews, B.W.** (1975). *Comparison of the Predicted and Observed Secondary Structure of T4 Phage Lysozyme.* Biochimica et Biophysica Acta, 405(2), 442–451.
    - Matthews Correlation Coefficient — balanced metric for imbalanced classification.

---

## Related Work (Multi-View Learning & Failure Prediction)

### Multi-View Learning

15. **Xu, C., Tao, D., & Xu, C.** (2013). *A Survey on Multi-View Learning.* arXiv:1304.5634.
    - Comprehensive survey on multi-view learning methods.
    - **Our use**: Our semantic grouping approach is a form of multi-view decomposition.

16. **Zhao, J., Xie, X., Xu, X., & Sun, S.** (2017). *Multi-View Learning Overview: Recent Progress and New Challenges.* Information Fusion, 38, 43–54.
    - Updated survey with deep multi-view learning methods.

### Predictive Maintenance

17. **Carvalho, T.P., Soares, F.A., Vita, R., Francisco, R.P., Basto, J.P., & Alcalá, S.G.** (2019). *A Systematic Literature Review of Machine Learning Methods Applied to Predictive Maintenance.* Computers & Industrial Engineering, 137, 106024.
    - Survey of ML methods for predictive maintenance, establishing baselines.

18. **Ran, Y., Zhou, X., Lin, P., Wen, Y., & Deng, R.** (2019). *A Survey of Predictive Maintenance: Systems, Purposes and Approaches.* arXiv:1911.07539.
    - Taxonomy of predictive maintenance approaches.

### AIOps & Distributed Systems Monitoring

19. **Notaro, P., Cardoso, J., & Gerndt, M.** (2021). *A Survey of AIOps Methods for Failure Management.* ACM Transactions on Intelligent Systems and Technology, 12(6), 1–45.
    - Survey of AI for IT operations, including failure detection in distributed systems.
    - **Our use**: Motivation for applying PID diagnostic to distributed systems.

20. **Soldani, J. & Brogi, A.** (2022). *Anomaly Detection and Failure Root Cause Analysis in (Micro)Service-Based Cloud Applications: A Survey.* ACM Computing Surveys, 55(3), 1–39.
    - Root cause analysis in microservices — our interpretability contribution is directly relevant.

---

## Baselines Cited

21. **Ali, M.J., Raza, B., Shahid, A.R., Mahmood, B., & Ignatious, H.A.** (2024). *Enhancing Software Defect Prediction Using Genetic Algorithm-Based Feature Selection and Ensemble Classifiers.* PeerJ Computer Science, 10, e1860. DOI: 10.7717/peerj-cs.1860
    - Original comparison baseline for the NASA PROMISE software defect prediction results.

22. **Chen, T. & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016, pp. 785–794.
    - XGBoost as the primary accuracy baseline in our comparisons.

---

## Future Work (Kafka Dataset)

23. **Kreps, J., Narkhede, N., & Rao, J.** (2011). *Kafka: A Distributed Messaging System for Log Processing.* NetDB Workshop.
    - Apache Kafka architecture and design principles.
    - **Our future work**: Build a real Kafka failure dataset with JMX/Prometheus instrumentation.

24. **Wang, P., Xu, J., Ma, M., Lin, W., Pan, D., Wang, Y., & Chen, P.** (2018). *CloudRanger: Root Cause Identification for Cloud Native Systems.* CCGRID 2018.
    - Methods for root cause analysis in cloud systems with cascading failures.
    - **Our future work**: Validate high-synergy hypothesis on real Kafka cascading failures.
