# User Guide
## GPU-Accelerated MLMC Network Modeling

This guide provides step-by-step instructions for using the framework.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Running Experiments](#running-experiments)
4. [Generating Visualizations](#generating-visualizations)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# 1. Navigate to project directory
cd GPU-Acc-Net-Prop-Congestion-Multi-Monte-Carlo

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
python -c "import sys; sys.path.insert(0, 'src'); from network.topology import TopologyGenerator; print('Installation successful!')"
```

### Quick Test

```bash
# Run a simple example
python examples/basic_simulation.py
```

---

## Basic Usage

### 1. Create a Network

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from network.topology import TopologyGenerator

# Initialize generator
gen = TopologyGenerator(seed=42)

# Create Erdős-Rényi random graph
network = gen.generate_erdos_renyi(n_nodes=100, p=0.15)

# Set link properties
network.set_link_properties(
    bandwidth_range=(1e9, 10e9),      # 1-10 Gbps
    delay_range=(0.001, 0.01),        # 1-10 ms
    capacity_range=(500, 2000),       # packets/sec
    seed=42
)

print(f"Network: {network.n_nodes} nodes, {network.n_edges} edges")
```

### 2. Define Traffic Model

```python
from network.traffic import PoissonTraffic, BurstyTraffic

# Option 1: Poisson traffic (simple)
traffic = PoissonTraffic(rate=100.0, seed=42)

# Option 2: Bursty traffic (realistic)
traffic = BurstyTraffic(
    on_rate=150.0,
    off_rate=50.0,
    burst_duration=2.0,
    idle_duration=0.5,
    burstiness=2.5,
    seed=42
)
```

### 3. Run Monte Carlo Simulation

```python
from simulation.monte_carlo import MonteCarloSimulator

# Create simulator
simulator = MonteCarloSimulator(seed=42)

# Run simulation
result = simulator.estimate(
    network=network,
    traffic=traffic,
    n_samples=1000,       # Number of MC samples
    T=10.0,              # Simulation time (seconds)
    dt=0.1,              # Timestep
    metric='mean_queue'  # Quantity of interest
)

# Print results
print(f"Mean: {result.mean:.4f}")
print(f"Std: {result.std:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### 4. Run MLMC Simulation

```python
from simulation.mlmc import MLMCSimulator

# Create MLMC simulator
mlmc_sim = MLMCSimulator(seed=42)

# Run MLMC
mlmc_result = mlmc_sim.mlmc_estimate(
    network=network,
    traffic=traffic,
    epsilon=0.05,        # Target accuracy
    L_max=4,             # Maximum levels
    T=10.0,              # Simulation time
    base_dt=0.2          # Base timestep
)

# Print results
print(f"Mean: {mlmc_result.mean:.4f}")
print(f"Levels: {mlmc_result.L}")
print(f"Samples per level: {mlmc_result.N_samples}")
print(f"Total cost: {mlmc_result.total_cost:.0f}")
```

### 5. Analyze Metrics

```python
from metrics.delay import DelayCalculator
from metrics.congestion import CongestionAnalyzer
from metrics.uncertainty import UncertaintyQuantifier

# Delay analysis
delay_calc = DelayCalculator(network, confidence_level=0.95)
# Generate delay samples (from simulation)
delay_samples = ... # Your delay data
delay_metrics = delay_calc.estimate_delay_distribution(delay_samples)

print(f"Mean delay: {delay_metrics.mean_delay*1000:.2f} ms")
print(f"P95 delay: {delay_metrics.percentiles['p95']*1000:.2f} ms")

# Congestion analysis
congestion_analyzer = CongestionAnalyzer(network, congestion_threshold=0.8)
queue_states = ...  # Your queue state data [n_timesteps, n_nodes]
congestion_metrics = congestion_analyzer.analyze_simulation_congestion(
    queue_states,
    arrival_rates=100.0,
    service_rates=120.0
)

print(f"Mean queue: {congestion_metrics.mean_queue_length:.2f}")
print(f"Congested nodes: {len(congestion_metrics.congested_nodes)}")

# Uncertainty quantification
uq = UncertaintyQuantifier(confidence_level=0.95, n_bootstrap=1000)
ci_lower, ci_upper, bootstrap_dist = uq.bootstrap_confidence_interval(delay_samples)

print(f"Bootstrap 95% CI: [{ci_lower*1000:.2f}, {ci_upper*1000:.2f}] ms")
```

---

## Running Experiments

### Experiment 1: MLMC Convergence

```bash
python experiments/exp1_mlmc_convergence.py
```

**What it does**:
- Validates MLMC convergence rate (MSE ∝ Cost⁻¹)
- Compares with standard MC (MSE ∝ Cost⁻⁰·⁵)
- Measures cost reduction (10x-100x)

**Output**:
- `results/tables/exp1_mlmc_convergence_results.json`
- `results/tables/exp1_cost_comparison.csv`

### Experiment 2: GPU Speedup

```bash
python experiments/exp2_gpu_speedup.py
```

**What it does**:
- Benchmarks GPU vs CPU performance
- Tests various sample sizes and network sizes
- Measures speedup (100x-500x expected)

**Output**:
- `results/tables/exp2_gpu_speedup_results.json`
- `results/tables/exp2_sample_size_scaling.csv`

**Note**: Requires PyCUDA. Falls back to CPU-only if GPU unavailable.

### Experiment 3: Uncertainty Quantification

```bash
python experiments/exp3_uncertainty_quantification.py
```

**What it does**:
- Demonstrates uncertainty-aware metrics
- Generates confidence intervals and uncertainty bands
- Compares stochastic vs deterministic predictions

**Output**:
- `results/tables/exp3_uncertainty_quantification_results.json`
- `results/tables/exp3_uncertainty_band.npz`

### Experiment 4: Real-World Validation

```bash
python experiments/exp4_realworld_validation.py
```

**What it does**:
- Validates on CAIDA AS topology
- Uses MAWI-based traffic model
- Identifies congestion hotspots

**Output**:
- `results/tables/exp4_realworld_validation_results.json`

**Note**: Uses synthetic fallback if CAIDA/MAWI data unavailable.

---

## Generating Visualizations

After running experiments, generate plots:

```bash
# Create scripts directory if needed
mkdir -p scripts

# Visualize all experiments
python scripts/visualize_results.py --experiment all

# Or specific experiments
python scripts/visualize_results.py --experiment 1 2
python scripts/visualize_results.py --experiment 3
```

**Generated Figures** (in `results/figures/`):
- `exp1_convergence_rate.png` - MSE vs Cost
- `exp1_variance_decay.png` - Level-wise variance
- `exp2_speedup_vs_samples.png` - GPU speedup scaling
- `exp2_speedup_vs_network_size.png` - Network size scaling
- `exp3_uncertainty_band.png` - Time series uncertainty
- `exp3_stochastic_vs_deterministic.png` - Comparison
- `exp4_network_scale.png` - Real-world scalability
- `exp4_prediction_uncertainty.png` - Confidence intervals

---

## Advanced Usage

### Using GPU Acceleration

```python
# Check GPU availability
try:
    from gpu.parallel_mc import GPUMonteCarloSimulator
    gpu_available = True
except ImportError:
    print("GPU not available. Install PyCUDA: pip install pycuda")
    gpu_available = False

if gpu_available:
    # Create GPU simulator
    gpu_sim = GPUMonteCarloSimulator(seed=42)

    # Run on GPU
    result = gpu_sim.estimate(
        network=network,
        traffic=traffic,
        n_samples=100000,  # Large sample count for GPU advantage
        T=10.0,
        dt=0.1
    )

    print(f"GPU Result: {result.mean:.4f}")
```

### Loading Real Datasets

```python
# SNAP dataset
from datasets.snap.loader import SNAPDatasetLoader

snap_loader = SNAPDatasetLoader()

# List available datasets
datasets = snap_loader.list_available_datasets()
for ds in datasets:
    print(f"{ds['name']}: {ds['nodes']} nodes")

# Load Email-Eu-core
network = snap_loader.load_dataset('email-Eu-core', download_if_missing=True)

# CAIDA AS topology
from datasets.caida.loader import CAIDATopologyLoader

caida_loader = CAIDATopologyLoader()
network = caida_loader.load_topology('20260101', as_undirected=True)

# MAWI traffic
from datasets.mawi.loader import MAWITraceProcessor

mawi_processor = MAWITraceProcessor()
stats = mawi_processor.extract_statistics_fast(pcap_path)
traffic = mawi_processor.create_traffic_model(stats=stats)
```

### Creating Synthetic Benchmarks

```python
from datasets.synthetic.generator import SyntheticBenchmarkGenerator

gen = SyntheticBenchmarkGenerator(seed=42)

# Stable queue scenario (M/M/1)
scenario = gen.generate_stable_queue_scenario(
    arrival_rate=8.0,
    service_rate=10.0,
    noise_intensity=0.5
)

network = scenario['network']
traffic = scenario['traffic']
ground_truth = scenario['ground_truth']

print(f"Expected queue length: {ground_truth['expected_queue_length']:.2f}")

# Convergence test suite
suite = gen.generate_convergence_test_suite()

for name, scenario in suite.items():
    print(f"{name}: utilization={scenario['ground_truth']['utilization']:.2f}")
```

---

## Troubleshooting

### Issue: Import errors

**Problem**: `ModuleNotFoundError: No module named 'network'`

**Solution**:
```python
# Add this at the top of your script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))  # Adjust path as needed
```

### Issue: GPU not available

**Problem**: `ImportError: No module named 'pycuda'`

**Solution**:
```bash
# Install PyCUDA
pip install pycuda

# If installation fails, check CUDA toolkit
nvidia-smi  # Verify GPU
nvcc --version  # Verify CUDA compiler

# Mac/Windows: GPU acceleration not supported (CUDA is NVIDIA-only)
# Use CPU-only mode
```

### Issue: Out of memory on GPU

**Problem**: `RuntimeError: out of memory`

**Solution**:
```python
# Reduce sample size or use batching
gpu_sim.estimate(
    network=network,
    traffic=traffic,
    n_samples=10000,  # Reduce from 1000000
    ...
)

# Or use CPU for very large problems
```

### Issue: Experiment results not found

**Problem**: `Results file not found`

**Solution**:
```bash
# Run the experiment first
python experiments/exp1_mlmc_convergence.py

# Then visualize
python scripts/visualize_results.py --experiment 1
```

### Issue: Dataset download fails

**Problem**: `Failed to download dataset`

**Solution**:
- Check internet connection
- For CAIDA/MAWI: May require registration (use synthetic fallback)
- Experiments work with synthetic data if real datasets unavailable

---

## Tips and Best Practices

1. **Start Small**: Test with small networks (50-100 nodes) before scaling up

2. **Use Synthetic Data**: Synthetic benchmarks are faster and have known ground truth

3. **GPU Advantage**: GPU shines for large sample counts (10⁴+); for small runs, CPU may be faster due to overhead

4. **MLMC Tuning**:
   - ε = 0.05 is a good starting point
   - L_max = 4-5 sufficient for most problems
   - base_dt = 0.1-0.2 works well

5. **Reproducibility**: Always set random seeds for reproducible results

6. **Logging**: Check console output for progress and warnings

---

## Next Steps

- Read the full [Project Report](docs/report/PROJECT_REPORT.md)
- Explore [Example Scripts](examples/)
- Run [Test Suite](tests/): `pytest tests/ -v`
- Customize for your network topology and traffic patterns
- Extend with new metrics or algorithms

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review example scripts in `examples/`
- Consult API documentation in source code docstrings

---

**Happy Simulating!**
