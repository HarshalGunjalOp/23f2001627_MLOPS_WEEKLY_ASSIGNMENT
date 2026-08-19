# Model Card: IRIS Classifier

* **Intended Use:** Classification of Iris flowers into three distinct species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.
* **Training Data:** The standard Fisher's Iris dataset (150 samples). A sensitive attribute, `location`, was generated but explicitly excluded from the training features to prevent spatial bias.
* **Performance & Fairness:** The model achieved 100% accuracy on the test set. Fairlearn audits confirm equitable performance across `location` subgroups (0 & 1), demonstrating no systemic demographic bias.
* **Limitations:** The model assumes the statistical distribution of flower measurements remains constant. It is susceptible to data drift if environmental factors systematically alter average flower dimensions in future data collections, which would require retraining.
