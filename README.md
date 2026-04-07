# AB Testing Platform

An interactive platform for understanding and implementing A/B testing concepts, from sample size calculation to business impact analysis.

## Architecture

```
ab_testing_platform/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── sample_size.py
│   ├── stat_tests.py
│   ├── simulations.py
│   ├── pitfalls.py
│   └── business_impact.py
│
├── tests/
│   └── test_sample_size.py # Unit tests
│
└── data/
    └── synthetic/          # Generated experiment data
```

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Modules

- **sample_size.py**: Calculate required sample size for experiments
- **stat_tests.py**: Statistical testing implementations
- **simulations.py**: Monte Carlo simulations for power analysis
- **pitfalls.py**: Common A/B testing pitfalls and solutions
- **business_impact.py**: Business metrics and impact analysis

## Testing

Run tests with:

```bash
pytest tests/
```
