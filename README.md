Project Title: Explainable AI in Healthcare Decision-Making for Breast Cancer Detection

1. Objective:
The objective of this project was to develop a deep learning-based breast cancer detection system using Convolutional Neural Networks (CNNs) while enhancing its interpretability using Explainable AI (XAI) techniques. We aimed to build a model that not only accurately classifies mammogram images as cancerous or non-cancerous but also provides explanations for its predictions, making AI-driven diagnosis more transparent, reliable, and trustworthy for clinicians.

2. Motivation:
Breast cancer is one of the most prevalent cancers worldwide, and early detection significantly increases survival rates. Deep learning models have demonstrated remarkable performance in medical image analysis but are often considered black boxes, meaning their decision-making process is opaque. This lack of explainability hinders their adoption in clinical settings. To address this, we leveraged XAI techniques like SHAP, LIME, Grad-CAM, and feature visualization to make the model’s predictions interpretable for healthcare professionals.

3. Dataset Information:
We used publicly available mammography datasets from Kaggle, which contained:

Over 54,706 images of mammograms.
Clinical metadata, including patient ID, age, cancer diagnosis, biopsy results, and imaging details.
The dataset included both cancerous (1) and non-cancerous (0) images, which were labeled accordingly.

4. Data Collection & Preprocessing:
Data Cleaning & Formatting: Extracted relevant information and removed unnecessary columns.

Image Processing: Selected 2,346 images (subset) for training.
Resized images to 512×512 pixels.
Normalized pixel values between 0 and 1.

Data Augmentation: To improve generalization, applied transformations such as rotation, flipping, and contrast adjustment.

Class Balancing: Ensured a balanced number of cancerous and non-cancerous images to avoid bias in model training.

5. Methodology & Algorithms: 
Model Architecture – CNN (Convolutional Neural Network)

We implemented a deep CNN model with the following architecture:
11 layers, including 3 convolutional layers, each followed by a max pooling layer to extract relevant features.
Dropout layers to prevent overfitting.
Dense layers at the end with ReLU activation for feature extraction and Softmax activation for classification.
Adam optimizer and SparseCategoricalCrossentropy loss function were used to train the model.

Explainable AI (XAI) Techniques Used. We applied four XAI techniques to improve model interpretability:
Feature Visualization – Analyzed which features (edges, shapes, textures) were most important in different CNN layers.
SHAP (SHapley Additive ExPlanations) – Identified pixel contributions to the model’s prediction.
LIME (Local Interpretable Model-Agnostic Explanations) – Generated perturbed versions of input images and analyzed how small changes influenced predictions.
Grad-CAM (Gradient-weighted Class Activation Mapping) – Produced heatmaps highlighting the most critical regions in an image that influenced the model’s decision.

6. Results & Interpretation:
CNN Performance:

Accuracy: ~92% on the test set.
Precision & Recall: Optimized to reduce false negatives, ensuring early cancer detection.
F1-score: Balanced between precision and recall, achieving high diagnostic performance.

Explainability Results:

Feature Visualization confirmed that the CNN effectively extracted important patterns from mammograms.
Grad-CAM provided clear and localized heatmaps, making it the most effective XAI technique.
SHAP and LIME were less effective due to unclear feature representation but still provided useful insights.

Final Output in a Web Application: Developed a Streamlit-based web app where users could upload a mammogram image. The system displayed cancer detection results along with a Grad-CAM heatmap, visually explaining the decision.

7. Challenges Faced:
Data Quality & Preprocessing: Handling low-resolution images and missing metadata was a challenge.
Model Interpretability: While Grad-CAM provided good visual explanations, other XAI methods (LIME, SHAP) were not always interpretable.
Computational Complexity: Training a deep CNN with explainability models required high GPU resources.

8. Future Scope:
Improving Model Robustness: Use pre-trained models (ResNet, EfficientNet) to improve accuracy.
Real-time Integration: Deploy the model as a clinical decision support tool for real-world usage.
Multimodal XAI: Combine different explainability techniques for a more comprehensive understanding.
Expansion to Other Diseases: Apply similar XAI-enhanced CNN models to other diseases like lung cancer and brain tumors.

9. Conclusion:
This project successfully demonstrated how Deep Learning and XAI can improve breast cancer detection and diagnosis. By integrating explainable AI techniques, we increased trust and adoption among healthcare professionals. Our CNN model, supported by Grad-CAM and feature visualization, provided both accurate predictions and meaningful explanations, making AI-driven diagnostics more interpretable and actionable in medical imaging.


