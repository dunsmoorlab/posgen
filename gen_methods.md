# Overview of (gus) Methods

## Curve Fitting

### Procedure

- Data are considered in tertiles, each subjects data is averaged for each 1/3 of generalization test
- we utilize scipy `optimize.curve_fit()` function to fit a binomial function to each subjects SCR data
    + Binomial is defined as `a*x**2 + b*x + c`
    + `optimize.curve_fit()` utilizes non-linear least squares to fit a function to data
- we then recover the parameters [a,b,c] from the line, and project it over the entire "emotion face space" (0-100%)

### Comparing Curves

- In order to compare the obtained estimated **generalization curves** (my proposal for what we call this analysis), we correlated responses across the entire face space between experiments
- In each bootstrap iteration, we resampled both the Posgen and Feargen data to collect 1 line per experiment.
- Then compute a pearson's correlation of the two **generalization curves**
    + n_boot iterations = 5000
- This allows us to estimate a 95% CI of the relationship between experiments
    + in the correlation graph, the red line represents the 95% CI, and the red dot represents the average correlation
    + everything was done in fisher z space, but put back in to Pearson's r for visualization
    + The shaded area around the lines in the curve plots represents the 95% CI for the lines

## Model comparison

- This section should go first in the paper, since it properly motivates the above analysis
- we computed linear and binomial fits of the data using the `lmer` package in r (mason should provide additional details here)
- In order to assess if a binomial model represented a better fit, we computed BIC (lower = better), as computed a log-liklihood ratio test test (citation needed)
    + the chi-square value here tests if a more complex model better explains the data
- found that for Posgen, except for the 1st tertile a more complex binomial better explains observed pattern of generlization than a 1st order model
    + not the case in feargen

Conclusions of all of this should be something to the effect of "Based on these analyses, we propose that this method be used to evaluate generalization in future experiments that utilize different classes of ecologically relevant stimuli"