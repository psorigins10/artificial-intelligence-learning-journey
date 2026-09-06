# R² Score

R² Score, also called the **Coefficient of Determination**, is a regression evaluation metric.

It measures how well a regression model explains the variation in the actual target values.

## Formula

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Where:

- **SS_res** = Sum of Squared Residuals
- **SS_tot** = Total Sum of Squares

## Interpretation

| R² Score | Meaning |
|---|---|
| 1 | Perfect predictions |
| 0 | Model is no better than predicting the mean |
| < 0 | Model performs worse than predicting the mean |

For example:

```text
R² = 0.85