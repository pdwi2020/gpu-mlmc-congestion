"""
Synthetic Benchmark Generator

Generate controlled synthetic datasets for testing and validation of
Monte Carlo and MLMC algorithms.

Provides:
- Controlled traffic scenarios with known ground truth
- Parameter sweeps for convergence testing
- Reproducible benchmarks
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from network.topology import NetworkGraph, TopologyGenerator
from network.traffic import PoissonTraffic, BurstyTraffic, TrafficModel
from network.sde import QueueDynamicsSDE

logger = logging.getLogger(__name__)


class SyntheticBenchmarkGenerator:
    """
    Generate synthetic benchmarks for MLMC validation.
    """

    def __init__(self, data_dir: Optional[Path] = None, seed: int = 42):
        """
        Initialize synthetic benchmark generator.

        Args:
            data_dir: Directory to store benchmarks (default: datasets/synthetic/)
            seed: Random seed for reproducibility
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        np.random.seed(seed)

        self.topology_gen = TopologyGenerator(seed=seed)

        logger.info(f"Synthetic benchmark directory: {self.data_dir}")

    def generate_stable_queue_scenario(self,
                                      arrival_rate: float = 8.0,
                                      service_rate: float = 10.0,
                                      noise_intensity: float = 0.5) -> Dict:
        """
        Generate stable queue scenario with known equilibrium.

        Args:
            arrival_rate: λ (packets/time)
            service_rate: μ (packets/time)
            noise_intensity: σ (noise coefficient)

        Returns:
            Scenario dictionary with network, traffic, and ground truth
        """
        # Verify stability: λ < μ
        if arrival_rate >= service_rate:
            raise ValueError(f"Queue is unstable: λ={arrival_rate} >= μ={service_rate}")

        # Create simple network (single queue)
        network = self.topology_gen.generate_erdos_renyi(n_nodes=10, p=0.3)
        network.set_link_properties(seed=self.seed)

        # Create traffic
        traffic = PoissonTraffic(rate=arrival_rate, seed=self.seed)

        # Ground truth (M/M/1 queue with noise)
        rho = arrival_rate / service_rate  # Utilization
        expected_queue = rho / (1 - rho)

        ground_truth = {
            'expected_queue_length': expected_queue,
            'utilization': rho,
            'stable': True,
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'noise_intensity': noise_intensity
        }

        scenario = {
            'name': 'stable_queue',
            'network': network,
            'traffic': traffic,
            'ground_truth': ground_truth,
            'parameters': {
                'arrival_rate': arrival_rate,
                'service_rate': service_rate,
                'noise_intensity': noise_intensity
            }
        }

        logger.info(f"Generated stable queue scenario: ρ={rho:.3f}, E[Q]={expected_queue:.3f}")

        return scenario

    def generate_congested_scenario(self,
                                   utilization: float = 0.95) -> Dict:
        """
        Generate highly congested scenario for stress testing.

        Args:
            utilization: Target utilization ρ (0.95 = 95% loaded)

        Returns:
            Scenario dictionary
        """
        service_rate = 10.0
        arrival_rate = utilization * service_rate

        # Higher noise for congested regime
        noise_intensity = 1.0

        network = self.topology_gen.generate_erdos_renyi(n_nodes=20, p=0.2)
        network.set_link_properties(seed=self.seed)

        traffic = PoissonTraffic(rate=arrival_rate, seed=self.seed)

        rho = arrival_rate / service_rate
        expected_queue = rho / (1 - rho)

        ground_truth = {
            'expected_queue_length': expected_queue,
            'utilization': rho,
            'stable': rho < 1.0,
            'congested': True
        }

        scenario = {
            'name': 'congested',
            'network': network,
            'traffic': traffic,
            'ground_truth': ground_truth,
            'parameters': {
                'utilization': utilization,
                'arrival_rate': arrival_rate,
                'service_rate': service_rate
            }
        }

        logger.info(f"Generated congested scenario: ρ={rho:.3f}, E[Q]={expected_queue:.1f}")

        return scenario

    def generate_bursty_traffic_scenario(self,
                                        burstiness: float = 3.0) -> Dict:
        """
        Generate scenario with bursty traffic.

        Args:
            burstiness: Burstiness coefficient (CV of inter-arrivals)

        Returns:
            Scenario dictionary
        """
        # Create bursty traffic with on-off model
        on_rate = 50.0
        mean_on_duration = 0.3
        mean_off_duration = 0.7

        network = self.topology_gen.generate_barabasi_albert(n_nodes=50, m=3)
        network.set_link_properties(seed=self.seed)

        traffic = BurstyTraffic(
            on_rate=on_rate,
            mean_on_duration=mean_on_duration,
            mean_off_duration=mean_off_duration,
            seed=self.seed
        )

        effective_rate = traffic.effective_rate

        scenario = {
            'name': 'bursty_traffic',
            'network': network,
            'traffic': traffic,
            'ground_truth': {
                'effective_rate': effective_rate,
                'burstiness': burstiness,
                'traffic_type': 'on_off'
            },
            'parameters': {
                'on_rate': on_rate,
                'mean_on_duration': mean_on_duration,
                'mean_off_duration': mean_off_duration
            }
        }

        logger.info(f"Generated bursty traffic scenario: effective rate={effective_rate:.2f}")

        return scenario

    def generate_convergence_test_suite(self) -> Dict[str, Dict]:
        """
        Generate suite of scenarios for MLMC convergence testing.

        Returns:
            Dictionary of scenarios with varying parameters
        """
        suite = {}

        # Scenario 1: Light load
        suite['light_load'] = self.generate_stable_queue_scenario(
            arrival_rate=5.0,
            service_rate=10.0,
            noise_intensity=0.2
        )

        # Scenario 2: Moderate load
        suite['moderate_load'] = self.generate_stable_queue_scenario(
            arrival_rate=8.0,
            service_rate=10.0,
            noise_intensity=0.5
        )

        # Scenario 3: Heavy load
        suite['heavy_load'] = self.generate_congested_scenario(
            utilization=0.9
        )

        # Scenario 4: Bursty traffic
        suite['bursty'] = self.generate_bursty_traffic_scenario(
            burstiness=2.5
        )

        logger.info(f"Generated convergence test suite with {len(suite)} scenarios")

        return suite

    def generate_parameter_sweep(self,
                                parameter: str,
                                values: List[float]) -> List[Dict]:
        """
        Generate scenarios for parameter sweep.

        Args:
            parameter: Parameter to sweep ('utilization', 'noise', 'burstiness')
            values: List of parameter values

        Returns:
            List of scenarios
        """
        scenarios = []

        for value in values:
            if parameter == 'utilization':
                scenario = self.generate_congested_scenario(utilization=value)

            elif parameter == 'noise':
                scenario = self.generate_stable_queue_scenario(
                    arrival_rate=8.0,
                    service_rate=10.0,
                    noise_intensity=value
                )

            elif parameter == 'burstiness':
                scenario = self.generate_bursty_traffic_scenario(burstiness=value)

            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            scenario['sweep_parameter'] = parameter
            scenario['sweep_value'] = value

            scenarios.append(scenario)

        logger.info(f"Generated parameter sweep: {parameter} with {len(scenarios)} values")

        return scenarios

    def save_scenario(self, scenario: Dict, filename: str):
        """
        Save scenario to file.

        Args:
            scenario: Scenario dictionary
            filename: Output filename (JSON)
        """
        filepath = self.data_dir / filename

        # Convert non-serializable objects to descriptions
        serializable = {
            'name': scenario['name'],
            'ground_truth': scenario['ground_truth'],
            'parameters': scenario['parameters'],
            'network': {
                'n_nodes': scenario['network'].n_nodes,
                'n_edges': scenario['network'].n_edges,
                'type': scenario['network'].__class__.__name__
            },
            'traffic': {
                'type': scenario['traffic'].__class__.__name__,
                'rate': getattr(scenario['traffic'], 'rate', None)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved scenario to {filepath}")

    def load_scenario(self, filename: str) -> Dict:
        """
        Load scenario from file.

        Args:
            filename: Input filename (JSON)

        Returns:
            Scenario metadata
        """
        filepath = self.data_dir / filename

        with open(filepath, 'r') as f:
            scenario = json.load(f)

        logger.info(f"Loaded scenario from {filepath}")

        return scenario

    def create_ground_truth_validation_set(self,
                                          n_samples: int = 10000,
                                          T: float = 100.0,
                                          dt: float = 0.01) -> Dict:
        """
        Create validation set with Monte Carlo ground truth.

        Generates high-precision MC estimates for validation.

        Args:
            n_samples: Number of MC samples (high for accuracy)
            T: Simulation duration
            dt: Time step (small for accuracy)

        Returns:
            Ground truth validation data
        """
        from simulation.monte_carlo import MonteCarloSimulator

        logger.info(f"Creating ground truth validation set: N={n_samples}, T={T}, dt={dt}")

        # Generate scenario
        scenario = self.generate_stable_queue_scenario()

        # Run high-precision MC
        simulator = MonteCarloSimulator(seed=self.seed)
        result = simulator.estimate(
            network=scenario['network'],
            traffic=scenario['traffic'],
            n_samples=n_samples,
            T=T,
            dt=dt,
            metric='mean_queue',
            verbose=False
        )

        validation_data = {
            'scenario': scenario['name'],
            'mc_estimate': result.mean,
            'mc_std': result.std,
            'mc_ci_lower': result.ci_lower,
            'mc_ci_upper': result.ci_upper,
            'n_samples': n_samples,
            'parameters': scenario['parameters'],
            'ground_truth_theoretical': scenario['ground_truth']
        }

        logger.info(f"Ground truth: MC={result.mean:.4f}±{result.std:.4f}, "
                   f"Theoretical={scenario['ground_truth']['expected_queue_length']:.4f}")

        return validation_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Synthetic Benchmark Generator - Example Usage")
    print("=" * 60)

    generator = SyntheticBenchmarkGenerator(seed=42)

    # Generate stable queue scenario
    print("\n1. Stable Queue Scenario")
    print("-" * 60)

    scenario = generator.generate_stable_queue_scenario(
        arrival_rate=8.0,
        service_rate=10.0,
        noise_intensity=0.5
    )

    print(f"Scenario: {scenario['name']}")
    print(f"Network: {scenario['network']}")
    print(f"Traffic: {scenario['traffic']}")
    print(f"\nGround Truth:")
    for key, value in scenario['ground_truth'].items():
        print(f"  {key}: {value}")

    # Generate convergence test suite
    print("\n2. Convergence Test Suite")
    print("-" * 60)

    suite = generator.generate_convergence_test_suite()

    print(f"Generated {len(suite)} test scenarios:")
    for name, scenario in suite.items():
        print(f"\n  {name}:")
        print(f"    Network: {scenario['network'].n_nodes} nodes")
        print(f"    Traffic: {scenario['traffic']}")
        if 'expected_queue_length' in scenario['ground_truth']:
            print(f"    Expected queue: {scenario['ground_truth']['expected_queue_length']:.3f}")

    # Generate parameter sweep
    print("\n3. Parameter Sweep")
    print("-" * 60)

    utilization_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sweep = generator.generate_parameter_sweep('utilization', utilization_values)

    print(f"Generated utilization sweep with {len(sweep)} scenarios:")
    for scenario in sweep:
        util = scenario['sweep_value']
        exp_q = scenario['ground_truth']['expected_queue_length']
        print(f"  ρ={util:.2f}: E[Q]={exp_q:.2f}")

    # Create ground truth validation
    print("\n4. Ground Truth Validation Set")
    print("-" * 60)

    print("Creating high-precision ground truth (this may take a moment)...")
    validation = generator.create_ground_truth_validation_set(
        n_samples=1000,  # Reduced for demo
        T=10.0,
        dt=0.1
    )

    print(f"\nValidation Data:")
    print(f"  MC Estimate: {validation['mc_estimate']:.4f} ± {validation['mc_std']:.4f}")
    print(f"  95% CI: [{validation['mc_ci_lower']:.4f}, {validation['mc_ci_upper']:.4f}]")
    print(f"  Theoretical: {validation['ground_truth_theoretical']['expected_queue_length']:.4f}")

    print("\n" + "=" * 60)
