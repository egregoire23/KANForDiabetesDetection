# Deep Learning for Diabetes Detection Using Kolmogorov-Arnold Networks
A Kolmogorov-Arnold Network (KAN) is a state-of-the-art deep learning model that increases the interpretability of a standard Multi-Layer Peceptron. In collaboration as a team of three, we constructed a KAN to evaluate its performance at identifying diabetes in patients, and provide research on how KANs will be invaluable in a healthcare setting.

Skills:

PyTorch, LLM, Hyperparameter Tuning, Handling Class Imbalance

Dataset:
- Diabetes Health Indicators Dataset ([link](www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

Methodology:
- Build, train, and evaluate our KAN from scratch
- Implement a standard MLP and the official KAN from library
- Provide a comparative analysis of our scratch KAN to the baselines
- Utilize an LLM to increase the interpretability of the results

Tools & Technologies:
- Programming Language: Python
- Libraries: torch, kagglehub, sklearn, pandas, matplotlib, openai
- Software: Jupyter Notebook

Results & Evaluation:

The KAN and MLP achieved very high performance scores, both reaching 86% accuracy and 0.3 loss on the test dataset. The KAN's added interpretability provided the value amount that each feature plays in determining whether a patient has diabetes or not. This found that the key indicators of diabetes include having heart disease/heart attack or the sex of the patient. While the KAN from library did not achieve as high a performance as the KAN from scratch or MLP, it still achieved a 73% accuracy and found that the most relevant features are stroke, physical activity, and income.

Challenges & Learning:

The biggest challenge in this project was learning how to adapt the weight vector of a standard MLP to instead be a vector of learnable activation functions. As a result, I learned new ways to visualize networks and their results, such as the activation functions and the model's architectures. Additionally, I learned how to integrate an LLM directly into a Jupyter Notebook for use alongside the project.

Contribution:
- Team Members: Erin Gregoire, Daniel Viola, & Dawson Damuth
- My Role: Build, train, and evaluate the Multi-Layer Perceptron and official KAN library
