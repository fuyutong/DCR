# Dynamic Class Reweighting-based Adversarial Robustness Enhancement for Medical Structured Data    


## Overview  
This repository implements **Dynamic Class Reweighting (DCR)**, an adversarial training framework designed to enhance robustness of medical AI models on structured clinical data (e.g., EHRs, lab results). The method addresses critical challenges in medical AI:  
-   **Class Imbalance**: Mitigates bias in adversarial sample generation for rare disease classes.  
-  🔒 **Worst-Class Robustness**: Ensures reliable performance under worst-case adversarial attacks.    
-   **Structured Data Compatibility**: Optimized for heterogeneous medical data (discrete codes + continuous features).  

Key innovations:  
1. **Balanced Adversarial Sample Generation**  
2. **Adaptive Loss Weighting via Effective Class Counts**  
3. **Minority-Class Feature Alignment Module**  
4. **Regret Minimization-based Worst-Class Attention**  

## Key Features  
-  ⚙️ **DCR Training Pipeline**: End-to-end adversarial training with configurable attack types (FGSM, PGD).    
-  📊 **Medical Dataset Support**: Preprocessed benchmarks for thyroid nodules, ulcerative colitis, and rare disease EHRs.  
-  📈 **Robustness Metrics**: Evaluates clean accuracy, worst-class robust accuracy, and attack transferability.  
