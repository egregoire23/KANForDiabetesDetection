# Deep Learning for Diabetes Detection Using Kolmogorov-Arnold Networks
A Kolmogorov-Arnold Network (KAN) is a state-of-the-art deep learning model that increases the interpretability of traditional Multi-Layer Perceptrons (MLPs).  In this project, our team of three designed and evaluated a custom KAN for detecting diabetes in patients. Our goal was to explore KAN’s performance and its potential impact in healthcare settings, where transparency and model explainability are critical.

Skills:

PyTorch, LLM, Hyperparameter Tuning, Handling Class Imbalance

Dataset:
- Diabetes Health Indicators Dataset ([link](www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset))

Methodology:
- Built, trained, and evaluated a KAN from scratch
- Implemented 2 baseline models: an MLP & the official KAN implementation from library
- Conducted comparative performance analysis across all models
- Utilize an LLM to enhance the interpretability of the results

Tools & Technologies:
- Programming Language: Python
- Libraries: torch, sklearn, pandas, matplotlib, openai, kagglehub,
- Software: Jupyter Notebook

Results & Evaluation:

The KAN and MLP achieved strong performance, both reaching 86% accuracy and 0.31 loss on the test dataset. The KAN's added interpretability revealed key feature importance for diabetes prediction. Top indicators include having heart disease or prior heart attack, or the sex of the patient. The official KAN library reached 73% accuracy and highlighted different influential features, such as stroke history, physical activity, and income level. These results demonstrate both the strength and flexibility of the KAN model in healthcare-related classification tasks.

Challenges & Learning:

A core challenge was modifying the weight vector of a standard MLP into a vector of learnable activation functions, a key component of KANs. As a result, I learned new ways to visualize networks and their results, such as the activation functions and the model's architectures. Additionally, I learned how to integrate an LLM directly into a Jupyter Notebook to assist in the explanation and analysis of model results.

Contribution:

Team Members: Erin Gregoire, Daniel Viola, & Dawson Damuth

My Role: 
- Built, trained, and evaluated the Multi-Layer Perceptron
- Implemented and tested the official KAN model from the public library
- Contributed to model evaluation and results interpretation
