# A/B Testing Intelligence Platform

An interactive Streamlit app for planning experiments, running statistical analysis, spotting common pitfalls, and translating experiment lift into business impact.

## Screenshots

Representative page snapshots:

### Experiment Planning

```text
+--------------------------------------------------+
| Experiment Planning                              |
|  - Sample Size Calculator                        |
|  - Test type / alpha / power inputs              |
|  - Per-group and total sample metrics            |
|  - Power Analysis curve with 80% crossover line  |
+--------------------------------------------------+
```

### Statistical Analysis

```text
+--------------------------------------------------+
| Statistical Analysis                             |
|  - CSV-driven test runner                        |
|  - P-value, test statistic, confidence interval  |
|  - P-value simulation histogram                  |
|  - Multiple-testing explainer                    |
+--------------------------------------------------+
```

### Pitfall Detection

```text
+--------------------------------------------------+
| Pitfall Detection                                |
|  - Simpson's paradox detector                    |
|  - Multiple-comparison correction table          |
|  - Peeking detector with alpha-crossing chart    |
+--------------------------------------------------+
```

### Business Impact

```text
+--------------------------------------------------+
| Business Impact                                  |
|  - ROI inputs via sliders and number fields      |
|  - Incremental / annualized revenue metrics      |
|  - Baseline vs projected revenue chart           |
|  - Executive markdown report generator           |
+--------------------------------------------------+
```

## Architecture

```text
project-root
|-- app.py
|-- requirements.txt
|-- README.md
|-- src
|   |-- sample_size.py
|   |-- stat_tests.py
|   |-- simulation.py
|   |-- corrections.py
|   |-- paradox_detector.py
|   |-- peeking_detector.py
|   |-- roi_calculator.py
|   `-- business_impact.py
`-- tests
    |-- test_app_pages.py
    |-- test_sample_size.py
    |-- test_stat_tests.py
    |-- test_simulation.py
    |-- test_corrections.py
    |-- test_paradox_detector.py
    |-- test_peeking_detector.py
    `-- test_roi_calculator.py
```

Runtime flow:

```text
Streamlit UI (app.py)
    -> planning inputs
        -> sample_size.py
    -> statistical tests and simulations
        -> stat_tests.py
        -> simulation.py
    -> pitfall checks
        -> corrections.py
        -> paradox_detector.py
        -> peeking_detector.py
    -> business impact and reporting
        -> roi_calculator.py
        -> business_impact.py
```

## Features

- `Experiment Planning`: sample-size calculator plus power analysis.
- `Statistical Analysis`: Welch t-test, chi-square, Mann-Whitney, and p-value simulation.
- `Pitfall Detection`: Simpson's paradox, multiple-comparison correction, and peeking-risk detection.
- `Business Impact`: ROI calculator, revenue projection chart, and executive markdown summary.

## How To Interpret Results

- A statistically significant result means the observed effect cleared your selected alpha threshold. It does not guarantee the effect is large, durable, or financially meaningful.
- Confidence intervals matter as much as the p-value. Narrow intervals suggest the estimate is stable; wide intervals suggest more uncertainty.
- A high peeking risk means significance appeared before the final planned readout, which increases false-positive risk unless a sequential-testing design was used.
- Multiple-comparison corrections help when you test many hypotheses at once. If a result disappears after correction, treat the original win cautiously.
- Positive ROI estimates are gross upside. Compare them against implementation cost, experiment overhead, and expected durability before shipping.
- `Run longer` is the default recommendation when the business case looks promising but the statistical evidence is still inconclusive.

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the app:

   ```bash
   streamlit run app.py
   ```

## Testing

Run the full test suite with:

```bash
pytest -q
```
