# Cell growth model

This cell growth model estimates the growth rate of cell cultures based on different parameters such as pH, temperature and other concentrations. The code compares different possible models and selects the most optimal one that performs the best for growth, therefore, the different parameters for growth 
are properly analysed and determined, and it can be clear which ones have the greatest impact on cell growth.

## Highlights

- Built pipeline with scikit-learn for two different regressions
- Optimized hyperparameters using cross validation
- Designed a reusable class to evaluate models on their accuracy
- Measured each variables effect with standardized coefficients
- Generated figures and tables using matplotlib to compare data

## Environment

- Python 3.14.7
- Matplotlib x.y.z
- Numpy 2.5.2
- pandas 3.0.5
- scikit-learn x.y.z

## Code Design

### Main.py

Running main.py executes everything that is in src, classe.py, constants.py and functions.py. main.py ensures that the functions called from those files are sent in the right order, and that the different functions to create tables and figures are all saved in their respective folder.

### Constants

To organize the project, all the constants like column names, number of folds, plot settings, etc.. are all set in here to ensure that they are all in the same place to avoid confusion. Keeping all the constants avoid having to look for them in other files, and changes can be made quickly in here to adjust for different models based on the settings that are to be studied.

###  Data

The cell growth dataset is interpreted using pandas, each column is chosen by it's name and if there are incomplete rows in the data, they are dropped so that there are no bugs in the model. Using this, the data is properly organized, and selected to ensure that no issues arise when building the model.

### Training and testing

20% of the data is used for testing and 80% for training, this ensures that the model is not biased towards some parts of the data, and that the model that it is built on can be tested on other data. Therefore, there is no bias towards the data that it was already trained on, so that new data can be properly evaluated unbiasely.

### Model pipelines

Each model is built with scikit-learn, with different pipelines for each model type, kNN or polynomial.
For polynomials, the pipeline starts by standardizing the values based on the range of each feature, then expands them into polynomial and interaction terms, and fits a linear regression.
Similarly, kNN standardizes the features, but this time applies the kNN regression. 
Standardizing is especially important for kNN because it compares distances for each feature, whereas for polynomials, weight can somewhat take into account the standardizing already.


### Hyperparamter tuning

Using the ModelCandidate class, and grid search, each polynomial function is evaluated from degrees 1 through 6 and k from 1 to 40, for both distance and uniform weighting. As previously mentioned, each model is tested on 80% of the data, so four folds, and then evaluated on the remaining fold (20%).
The scores are based on RMSE, and that's the way they are compared in the tables and figures.

### Model Selection

Each model is evaluated on RMSE specifically from the cross-validated one, so, the RMSE is calculated for all the different folds possible as the training test, to have less bias.
If models have similar RMSE, the simpler one is selected since simplicity is also important.

### Interpretation

Polynomial models can be interpreted by their weights easily, since they have been standardized. Therefore, it is really easy to compare each weight to understand which settings have more impact on cell growth.
All of these terms are stored in a table and sorted by magnitude to compare their effects.
Also, the best cell growth conditions and the worst are also put into a table to understand what are the best ranges for features, without being standardized.

### Learning curve

The models are re-tuned and re-evaluated on progressively larger random subsets of the data. The resulting learning curve shows whether collecting more samples would be likely to improve accuracy, which helps decide whether the model's limitations come from too little data or from noise in the measurements themselves.

## Analysis


### Dataset

The dataset contains 318 samples in total, with respective ranges of 30 to 42 degrees Celsius for temperature, 6 to 7.8 for pH, glucose and lactate concentration of 0 to 40 mM. Also growth rate ranges from 0 to 0.75 d^-1.


### Choice of model

The selected model is a second-order polynomial regression with interaction terms: the four standardized features are expanded into 14 terms (4 linear, 4 squared, 6 pairwise interactions) and fitted by ordinary least squares.

| Degree | Train RMSE [d⁻¹] | Validation RMSE [d⁻¹] |
|-------:|-----------------:|----------------------:|
| 1 | 0.171 | 0.177 |
| **2** | **0.143** | **0.157** |
| 3 | 0.130 | 0.171 |
| 4 | 0.112 | 0.235 |
| 5 | 0.086 | 0.618 |
| 6 | 0.000 | 20.53 |

Clearly, as the polynomial increase, teh train RMSE decrease, but the validation RMSE increases by a lot, which is what matters. Hence, the choice of second order polynomial, since the validation RMSE is at it's lowest.
Not to mention, the second order is reasonably simple to interpret, and it only allows for interactions between two features, which is more likely than interaction of 4 or more features for example.
Therefore, only the important interactions are weighted, making this model better.

### Comparing with k-NN

in table_03, k with 10 distance based falls within the a similar cross validation RMSE, as demonstrated below.

| Model | CV RMSE [d⁻¹] | Train RMSE [d⁻¹] | Test RMSE [d⁻¹] | Test R² |
|-------|--------------:|-----------------:|----------------:|--------:|
| **Polynomial (degree = 2)** | **0.157 ± 0.016** | 0.144 | **0.157** | **0.38** |
| kNN (k = 10, distance) | 0.155 ± 0.011 | 0.000 | 0.156 | 0.39 |

Here, even though the knn model has better CV RMSE, the polynomial is still considered the better model, since distance based knn regression are usually more prone to overfitting, since closer values are often way too heavily weighted. Also, polynomial is more easily interpretable, so it is a better model.


### Advantages 

- The model is interpretable: its weights show the direction and relative strength of each effect.
- It is compact and fast, needing only 15 coefficients to make a prediction.
- It produces smooth predictions, unlike the step-like output of kNN.
- It generalizes as well as the more flexible kNN model while being far simpler.

### Limitations

- The accuracy of the model isn't ideal, with a value of R^2 of 0.38, it means that the model only accounts for 38% of the variation in growth rate, and the other 62% is unexplained
- The mean growth rate is 0.28 d^-1, with a prediction error of 0.16 d^-1, so the unpredictability likely comes from the small data range itself, or some other unmeasured variables that could have an impact on the cell growth.
- Model doesn't know the possible range of cell growth, like negative values or values higher than the maximum, so some values predicted can be physically impossible, if near the edges of the data.
- Second order model cannot account for sharper impacts (overfitting almost), which can cause inaccuracies if some specific ranges of features are highly sensitive at one certain value.

Looking at the learning curve, the model doesn't get better as more data is introduced, therefore indicating that it's likely the noise in the data itself limiting the RMSE at that value, and not the model.
### Variable effects

Effect of each variable

The figure varies one variable across its observed range while holding the others at their mean values. The largest polynomial weights (on standardized features) quantify these effects:

| Term | Weight |
|------|-------:|
| Intercept (baseline) | 0.338 |
| [pH]² | −0.061 |
| [Lactate] | −0.060 |
| [Glucose] | +0.025 |
| [Glucose] × [Lactate] | −0.021 |
| [Temperature]² | −0.021 |
| [Temperature] × [pH] | −0.020 |

- pH has the strongest effect. The large negative squared term creates a clear optimum near pH 6.9 to 7.0, with growth falling steeply on both sides. Because the effect is symmetric, pH shows almost no linear correlation with growth rate, which is exactly why the linear model underfits.
- Lactate is nearly as important and inhibits growth: growth rate falls from about 0.47 d⁻¹ with no lactate to about 0.28 d⁻¹ near 30 mM. The small positive squared term means the inhibition levels off at high concentrations.
- Glucose increases growth, from about 0.30 to 0.44 d⁻¹ across its range. The negative glucose-lactate interaction shows that the benefit of glucose is smaller when lactate has accumulated
- Temperature has a mild optimum around 36 °C. The negative temperature-pH interaction means that high temperature and high pH together are especially unfavourable.

The kNN model, shown in the same figure, traces the same overall shapes, which increases confidence that these effects are real features of the data rather than artifacts of the polynomial form.

### Optimal conditions

Within the range covered by the data, the highest predicted growth rates (about 0.57 to 0.58 d⁻¹) occur at:

- Temperature of about 36 to 37 degreees Celsius
- pH of about 6.85 to 7.05
- glucose of about 29 mM, the top of the searched range
- no lactate

In practice, this corresponds to a culture kept near its temperature and pH optimum, with ample glucose and lactate removed or kept from accumulating.

### Clear poor conditions

The lowest predicted growth rates occur at about 40 °C, pH 7.5, and lactate of 28 to 31 mM, where the model predicts essentially zero growth (slightly negative values of −0.04 to −0.07 d⁻¹, which should be read as zero). Growth also falls sharply at low pH: below about pH 6.3, the predicted growth rate drops toward zero even when the other variables are at typical values.


### Unreliable predictions from model

- Outside the observed ranges, the model can quickly become inaccurate since it has be trained on data within that range, especially since the model does physically limit any values, which can make cell growth predictions quite unrealistic.
- Near the edges of the observed ranges, where samples are sparse: below about 32 °C, above 41 °C, below pH 6.3, above pH 7.6, and above about 33 mM glucose. At pH 6.0, for example, the polynomial predicts a negative growth rate of about −0.13 d^-1.
- Combinations of extreme values, such as low temperature together with low pH. Each value may lie within its own observed range, but the combination never occurs in the dataset.
- Near zero growth or above about 0.6 d^-1, where the measured values appear to be bounded and the model cannot represent the cut-off.


### Useful applications

- In practice, it helps determine some baseline optimal conditions, clearly, some pH values and temperatures are clearly superior than others regardless, and should be kept within certain ranges no matter what.
- Lactate is shown as an inhibitor, so controlling how to remove it from the reactor can really have an impact on cell growth and should taken into consideration
- It shows that the most important features, like pH and nutrient concentrations should be monitored closer, and probably with more sensitivity than the other features.




