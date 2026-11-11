## Vis4AI

Phase I: How does the model behave?

In this phase, you will treat the model as a black box; you don’t look inside it. You only look at inputs and outputs and build up a picture of its behavior from that alone. To this end, you need a dataset of inputs from which you can obtain model outputs. This dataset can be:

- an official test set released with the model

- an established benchmark for the same task

- a synthetic dataset designed by you to investigate particular edge cases or hypotheses

The dataset needs to be large enough that any pattern you report can be shown to be systematic behavior, not a handful of cherry-picked examples.

What counts as “behavior” can include specific error patterns, biases, hallucinations, or shifts in output characteristics like diversity, confidence, style, etc.

If you have access to the training data, you can also explore it as a source of hypotheses. For example, exploring the training data might show that a class is underrepresented; you would then go back to the model outputs to confirm or disconfirm that this shows up as biased behavior.

Note that we are not aiming for a model performance report; a table of benchmark scores is not sufficient. Your visualization should expose behaviors that cannot be understood from aggregate performance measures alone. Additionally, simply identifying a pattern is not sufficient; you are supposed to understand it in depth. For example, if you have a 2D projection of inputs, discovering a cluster for which the model makes many incorrect predictions is not enough. You are expected to ask further questions like: What distinguishes this cluster from other clusters? Why is the cluster particularly dense or sparse? Does it have sub-clusters where the model behaves differently? etc.

this is our task, we are going to use tirex2 model and use time benchmark dataset with 39 domains with different frequencies. can u gives us ideas how to approach this. we already have results for long, medium and short horizons and extracted various features of each time series.

Answer:

H1. Horizon sensitivity

Tirex2 may behave differently as the forecast horizon increases. Does it become systematically smoother, more biased, or less responsive to sudden changes?

H2. Frequency sensitivity

Does Tirex2 struggle more with certain sampling frequencies? For example, does it capture daily patterns better than hourly patterns?

H3. Domain-specific behavior

Are there domains in which Tirex2 consistently makes particular types of mistakes? Are these domains actually similar in their underlying time-series characteristics?

H4. Input characteristic sensitivity

Which time-series characteristics are associated with particular model behaviors? For example, does high seasonality, intermittency, trend strength, or volatility affect forecasting behavior?

H5. Consistency and robustness

Does Tirex2 behave consistently across time series with similar characteristics, or does it produce very different forecasts for apparently similar inputs?

2. Go beyond forecasting errors: extract behavioral indicators

3. Investigate interactions between the three dimensions

This is where your project can become much more interesting.

Instead of analyzing domain, frequency, horizon, and features separately, investigate their interactions.

For example:

A. Does the long horizon affect all frequencies equally?

Create a heatmap:

\- Rows: 39 domains.

\- Columns: short, medium, long horizons.

\- Color: amplitude ratio, signed bias, or normalized forecasting error.

C. Does the model behave differently for similar time series across different domains?

This is one of the most interesting questions you can ask.

Imagine two time series:

\- One from the energy domain.

\- One from the traffic domain.

Both have:

\- Strong daily seasonality.

- \- Similar trend strength. - Similar volatility.

\- Similar autocorrelation.

Do their forecasts exhibit similar behavior?

If not, you have evidence that domain membership is associated with differences that your extracted features do not explain.

This is particularly interesting for a general-purpose forecasting model.

. Use dimensionality reduction to discover behavioral clusters

Since you already extracted various time-series features, I would definitely try this.

The idea is to create a representation of your dataset where time series with similar characteristics are close together.

can i compare seasonal strength how it relates to MASE

ANSWER: Yes! In fact, seasonal strength vs. MASE is one of the best analyses you could do for your Vis4AI project.

It allows you to investigate whether Tirex2's forecasting errors are systematically related to how seasonal a time series is.

What else could i do? Since you already have Tirex2 predictions for three horizons, 39 domains, different frequencies, and extracted time-series features, you can investigate many more interesting relationships.

## QUESTIONS:

## QUESTIONS:


My suggestion is to shift your thinking from:

"Which time series does Tirex2 forecast poorly?"

to:

"What characteristics make Tirex2 behave in a particular way, and how does that behavior change?"

Here are 10 experiments I think would be particularly valuable for your Vis4AI project. 1. Trend strength vs. forecasting error High priority Research question: Does Tirex2 perform better on strongly trending time series? - X-axis: Trend strength - Y-axis: MASE - Color: Forecast horizon Possible discoveries: - Strong trends are easier to extrapolate. - Strong trends cause larger forecasting errors. - Tirex2 struggles with changing trends. - Trend strength matters only for long horizons. Go one step further: distinguish between increasing and decreasing trends. Does the model have a systematic directional bias? 2. Entropy vs. forecasting error Research question: Does Tirex2 struggle with unpredictable or complex time series? - X-axis: Spectral entropy or sample entropy. - Y-axis: MASE. - Color: Frequency or horizon.

Question: Prompt(1) I have this project: "Vis4AI Phase I: How does the model behave? In this phase, you will treat the model as a black box; you don’t look inside it. You only look at inputs and outputs and build up a picture of its behavior from that alone. To this end, you need a dataset of inputs from which you can obtain model outputs. This

dataset can be:

- an official test set released with the model

- an established benchmark for the same task

- a synthetic dataset designed by you to investigate particular edge cases or hypotheses

The dataset needs to be large enough that any pattern you report can be shown to be systematic behavior, not a handful of cherry-picked examples. What counts as “behavior” can include specific error patterns, biases, hallucinations, or shifts in output characteristics like diversity, confidence, style, etc. If you have access to the training data, you can also explore it as a source of hypotheses. For example, exploring the training data might show that a class is underrepresented; you would then go back to the model outputs to confirm or disconfirm that this shows up as biased behavior. Note that we are not aiming for a model performance report; a table of benchmark scores is not sufficient. Your visualization should expose behaviors that cannot be understood from aggregate performance measures alone. Additionally, simply identifying a pattern is not sufficient; you are supposed to understand it in depth. For example, if you have a 2D projection of inputs, discovering a cluster for which the model makes many incorrect predictions is not enough. You are expected to ask further questions like: What distinguishes this cluster from other clusters? Why is the cluster particularly dense or sparse? Does it have sub-clusters where the model behaves differently? etc.". And I need you to help me brainstorm ideas. I am using the TIME benchmark datasets from Hugging Face and the TiRex2 model. I want to investigate biases that the model might have. For investigating, I was thinking of comparing datasets from the same category and then different categories to see if the model behaves a certain way within a category or overall. I also have different frequencies for some datasets and I think checking the difference between them would be interesting too. What do you think? How would you investigate this? Would you do something else? Here is also some of my code: and its output:

Answer: Your plan (within category, across categories, across frequencies) is a good skeleton. I’d add three things to it: fix a few issues in the current pipeline, make comparisons that control for confounds, and add experiments where you control the input, since those turn “I noticed a pattern” into “the model does this.”

## Issues in the current run


- 1. Your sample is very unbalanced. Azure alone contributes about 1,500 of the 1,953 records, and most datasets have 1 sample. Any pooled number is essentially “TiRex-2 on Azure.” Aggregate per dataset first, then compare datasets, and use bootstrap confidence intervals.

- 2. 433 samples were skipped, probably not at random. The skip policy drops anything containing a NaN. The progress log suggests entire datasets (e.g. SG_Carpark with 354 samples, Smart_Manufacturing, Water_Quality) never produced a record. That is survivorship bias: you only study the clean domains. Switch to interpolate, or at least log which datasets were dropped and why.

- 3. One window per series is too thin. With a single last-20% window, many datasets have n=1 forecast, so you can’t call anything “systematic.” Use rolling-origin evaluation (e.g. 5–20 forecast origins per series) so each dataset yields many forecasts.

- 4. The horizon is confounded with frequency. A horizon of up to 320 steps at 5T is about 27 hours, while at M it is over 26 years. For frequency comparisons, fix the horizon and context length in steps (and separately in wall-clock time where possible).

- 5. You have no baseline. “The model is biased on energy data” can’t be separated from “energy data is hard.” Add a seasonal-naive forecast (using the period implied by the frequency) and report the model’s error relative to it. Your current MASE uses a lag-1 naive scale, which is poor for strongly seasonal data. Also scale residuals by context scale so they can be compared across datasets.

## Behaviors worth investigating

## Forecast shape (the black-box behaviors that MAE hides)

- Trend damping and mean reversion: compare forecast slope to actual slope, split by trend direction. Do upward and downward trends get treated symmetrically?

- Amplitude decay: does seasonal amplitude shrink along the horizon? Does the phase drift? Plot error and volatility ratio as a function of horizon step, not just averages.

- Flat-line and collapse behavior: which series characteristics trigger it (intermittent, noisy, short context)?

- Level-shift lag: after a regime change near the end of the context, how long until the forecast catches up?

## Calibration

- Coverage by horizon step, domain, and frequency, plus PIT histograms.

- Asymmetry: does it miss more above or below? Is it overconfident on noisy data and underconfident on smooth data?

- Sharpness vs. error: is interval width informative about actual error?

## Multivariate behavior (your data is 96% multivariate)

- Forecast jointly vs. each variate independently. Does joint modeling help, hurt, or change the shape of the forecasts?

- Does error depend on the number of variates? Are residuals correlated across variates (shared failure) or independent?

## Natural experiments in your data

You have the same phenomenon at multiple sampling rates: current_velocity (5T/10T/15T/20T/H), Coastal_T_S (5T/15T/20T/H), CPHL (15T/30T/H), plus Port_Activity and ECDC_COVID at D vs W. These are paired comparisons, which is much stronger than comparing unrelated datasets.

You can also create your own: resample one high-frequency series to several coarser frequencies and compare the same signal at each. The question is whether the model’s behavior (damping, calibration, seasonality capture) depends on the sampling rate itself, or on what the rate does to the series (period length in steps, noise, effective context).

## Controlled experiments

These are cheap and expose biases cleanly:


- Invariance tests: multiply by 1000, add a large offset, flip the sign, reverse time. A well-behaved model should handle these predictably, and sign-flip asymmetry is a direct test of directional bias.

- Context-length sweep: feed 32, 64, 128, 512, and 2048 steps of the same series. Where does behavior stabilize?

- Synthetic probes: sine at various periods, trend + season, step changes, spikes, intermittent series, white noise, random walk. Vary period relative to context length. These test specific hypotheses (e.g. “the model fails when the period exceeds N steps”).

- Injected outliers or missing blocks in the context, to see how far the forecast is dragged.

## Within vs. across category

Category is often a proxy for series characteristics. To separate the two:

- 1. Compute per-series features: trend strength, seasonal strength and period, spectral entropy, noise level, intermittency/zero fraction, stationarity, outlier rate, context length, number of variates.

- 2. Project the features (UMAP or PCA) and color by relative error. Then drill into clusters, as the assignment asks: what defines a bad cluster, does it have sub-clusters, and is it separated by domain, frequency, or just by features?

- 3. Ask how much error variance is explained by domain vs. frequency vs. features (a mixed-effects model with dataset as a random effect works well). If domain adds nothing once features are included, the “category bias” is really a “characteristics bias,” which is itself a good finding.

## Visualization ideas

- A horizon × dataset heatmap of signed error or volatility ratio.

- Small multiples of forecasts, sorted or clustered by residual pattern.

- A linked view: the feature-space projection on the left, and the selected series’ context, forecast, and quantile band on the right.

- PIT or coverage-by-horizon curves, faceted by frequency.

## What I’d do first

- 1. Fix the pipeline: interpolate NaNs, use rolling origins, add the seasonal-naive baseline, and record frequency and category per record.

- 2. Run the natural frequency experiments, since they are the most distinctive part of your project.

- 3. Extract series features and build the feature-space view.

- 4. Run invariance and synthetic probes to confirm the hypotheses the first three steps suggest.

One caveat to mention in your write-up: if TiRex-2’s training data overlapped with TIME datasets, some “good” behavior could reflect memorization rather than generalization. It’s worth checking the TIME paper and the model card for this.

Do you want me to sketch the code changes for rolling-origin evaluation and feature extraction first?

Question: Prompt(2) I am using the TIME benchmark datasets and I trained them using the TiRex2 model. I need to analyze and visualize bias, hallucination made by the model. I have the results, features and a data_cube. My results file looks like this: output/results/TiRex2/<dataset>/<freq>/<term>. So far I analyzed Qunatile Crossing and Uncertainty Explosion using this code:. Is the code correct? Considering my datasets have different lengths. What should I add to investigate further and have more visualizations, not only code. I would go further deep to plot 1 dataset with mean lag1 close to 1, another one close to 0 and one in the middle.

Question: Prompt(3) Here is my task: …. Which plots do you think would be helpful? Please provide me with code! Question: Prompt(4) Here is my task: … and here is what I have so far: … Please act as my teacher and provide me feedback. What do you think I should add to further investigate my findings? Please provide code!
