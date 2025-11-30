#!/usr/bin/env python3
"""
TSPLIB Benchmark for Clock-Resonant Optimizer v2
=================================================

Tests the optimizer on standard TSPLIB instances to validate
real-world performance.

TSPLIB instances are downloaded from:
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

We'll use a few classic instances:
- eil51 (51 cities, optimal = 426)
- berlin52 (52 cities, optimal = 7542)
- st70 (70 cities, optimal = 675)
- eil76 (76 cities, optimal = 538)
- kroA100 (100 cities, optimal = 21282)
"""

import numpy as np
import time
import sys
import os
import urllib.request
import re

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from workbench.processors.sublinear_qik import SublinearQIK
from workbench.processors.sublinear_clock import SublinearClockOptimizer
from workbench.processors.sublinear_clock_v2 import SublinearClockOptimizerV2


# TSPLIB instance data (embedded to avoid download issues)
TSPLIB_INSTANCES = {
    'eil51': {
        'optimal': 426,
        'coords': [
            (37, 52), (49, 49), (52, 64), (20, 26), (40, 30), (21, 47), (17, 63), (31, 62),
            (52, 33), (51, 21), (42, 41), (31, 32), (5, 25), (12, 42), (36, 16), (52, 41),
            (27, 23), (17, 33), (13, 13), (57, 58), (62, 42), (42, 57), (16, 57), (8, 52),
            (7, 38), (27, 68), (30, 48), (43, 67), (58, 48), (58, 27), (37, 69), (38, 46),
            (46, 10), (61, 33), (62, 63), (63, 69), (32, 22), (45, 35), (59, 15), (5, 6),
            (10, 17), (21, 10), (5, 64), (30, 15), (39, 10), (32, 39), (25, 32), (25, 55),
            (48, 28), (56, 37), (30, 40)
        ]
    },
    'berlin52': {
        'optimal': 7542,
        'coords': [
            (565, 575), (25, 185), (345, 750), (945, 685), (845, 655), (880, 660), (25, 230),
            (525, 1000), (580, 1175), (650, 1130), (1605, 620), (1220, 580), (1465, 200),
            (1530, 5), (845, 680), (725, 370), (145, 665), (415, 635), (510, 875), (560, 365),
            (300, 465), (520, 585), (480, 415), (835, 625), (975, 580), (1215, 245), (1320, 315),
            (1250, 400), (660, 180), (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580),
            (685, 595), (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960),
            (95, 260), (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65), (830, 610),
            (605, 625), (595, 360), (1340, 725), (1740, 245)
        ]
    },
    'st70': {
        'optimal': 675,
        'coords': [
            (64, 96), (80, 39), (69, 23), (72, 42), (48, 67), (58, 43), (81, 34), (79, 17),
            (30, 23), (42, 67), (7, 76), (29, 51), (78, 92), (64, 8), (95, 57), (57, 91),
            (40, 35), (68, 40), (92, 34), (62, 1), (28, 43), (76, 73), (67, 88), (93, 54),
            (6, 8), (87, 18), (30, 9), (77, 13), (78, 94), (55, 3), (82, 88), (73, 28),
            (20, 55), (27, 43), (95, 86), (67, 99), (48, 83), (75, 81), (8, 19), (20, 18),
            (54, 38), (63, 36), (44, 33), (52, 18), (12, 13), (25, 5), (58, 85), (5, 67),
            (90, 9), (41, 76), (25, 76), (37, 64), (56, 63), (10, 55), (98, 7), (16, 74),
            (89, 60), (48, 82), (81, 76), (29, 60), (17, 22), (5, 45), (79, 70), (9, 100),
            (17, 82), (74, 67), (10, 68), (48, 19), (83, 86), (84, 94)
        ]
    },
    'eil76': {
        'optimal': 538,
        'coords': [
            (22, 22), (36, 26), (21, 45), (45, 35), (55, 20), (33, 34), (50, 50), (55, 45),
            (26, 59), (40, 66), (55, 65), (35, 51), (62, 35), (62, 57), (62, 24), (21, 36),
            (33, 44), (9, 56), (62, 48), (66, 14), (44, 13), (26, 13), (11, 28), (7, 43),
            (17, 64), (41, 46), (55, 34), (35, 16), (52, 26), (43, 26), (31, 76), (22, 53),
            (26, 29), (50, 40), (55, 50), (54, 10), (60, 15), (47, 66), (30, 60), (30, 50),
            (12, 17), (15, 14), (16, 19), (21, 48), (50, 30), (51, 42), (50, 15), (48, 21),
            (12, 38), (15, 56), (29, 39), (54, 38), (55, 57), (67, 41), (10, 70), (6, 25),
            (65, 27), (40, 60), (70, 64), (64, 4), (36, 6), (30, 20), (20, 30), (15, 5),
            (50, 70), (57, 72), (45, 42), (38, 33), (50, 4), (66, 8), (59, 5), (35, 60),
            (27, 24), (40, 20), (40, 37), (40, 40)
        ]
    },
    'kroA100': {
        'optimal': 21282,
        'coords': [
            (1380, 939), (2848, 96), (3510, 1671), (457, 334), (3888, 666), (984, 965),
            (2721, 1482), (1286, 525), (2716, 1432), (738, 1325), (1251, 1832), (2728, 1698),
            (3815, 169), (3683, 1533), (1247, 1945), (123, 862), (1234, 1946), (252, 1240),
            (611, 673), (2576, 1676), (928, 1700), (53, 857), (1807, 1711), (274, 1420),
            (2574, 946), (178, 24), (2678, 1825), (1795, 962), (3384, 1498), (3520, 1079),
            (1256, 61), (1424, 1728), (3913, 192), (3085, 1528), (2573, 1969), (463, 1670),
            (3875, 598), (298, 1513), (3479, 821), (2542, 236), (3955, 1743), (1323, 280),
            (3447, 1830), (2936, 337), (1621, 1830), (3373, 1646), (1393, 1368), (3874, 1318),
            (938, 955), (3022, 474), (2482, 1183), (3854, 923), (376, 825), (2519, 135),
            (2945, 1622), (953, 268), (2628, 1479), (2097, 981), (890, 1846), (2139, 1806),
            (2421, 1007), (2290, 1810), (1115, 1052), (2588, 302), (327, 265), (241, 341),
            (1917, 687), (2991, 792), (2573, 599), (19, 674), (3911, 1673), (872, 1559),
            (2863, 558), (929, 1766), (839, 620), (3893, 102), (2178, 1619), (3822, 899),
            (378, 1048), (1178, 100), (2599, 901), (3416, 143), (2961, 1605), (611, 1384),
            (3113, 885), (2597, 1830), (2586, 1286), (161, 906), (1429, 134), (742, 1025),
            (1625, 1651), (1187, 706), (1787, 1009), (22, 987), (3640, 43), (3756, 882),
            (776, 392), (1724, 1642), (198, 1810), (3950, 1558)
        ]
    }
}


def load_tsplib_instance(name: str) -> tuple:
    """Load a TSPLIB instance."""
    if name not in TSPLIB_INSTANCES:
        raise ValueError(f"Unknown instance: {name}")
    
    instance = TSPLIB_INSTANCES[name]
    coords = np.array(instance['coords'], dtype=float)
    optimal = instance['optimal']
    
    return coords, optimal


def benchmark_instance(name: str, n_trials: int = 3):
    """Benchmark all optimizers on a single instance."""
    coords, optimal = load_tsplib_instance(name)
    n = len(coords)
    
    print(f"\n{'='*70}")
    print(f"Instance: {name} (n={n}, optimal={optimal})")
    print(f"{'='*70}")
    
    results = {}
    
    # Original SublinearQIK
    original = SublinearQIK()
    lengths = []
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, _ = original.optimize_tsp(coords)
        times.append(time.time() - t0)
        lengths.append(length)
    results['original'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - optimal) / optimal
    }
    
    # Clock v1
    clock_v1 = SublinearClockOptimizer(use_multi_clock=True)
    lengths = []
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, _ = clock_v1.optimize_tsp(coords)
        times.append(time.time() - t0)
        lengths.append(length)
    results['clock_v1'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - optimal) / optimal
    }
    
    # Clock v2 (full features)
    clock_v2 = SublinearClockOptimizerV2(
        use_6d_tensor=True,
        use_gradient_flow=True,
        use_adaptive_dimension=True
    )
    lengths = []
    times = []
    stats_list = []
    for _ in range(n_trials):
        t0 = time.time()
        tour, length, stats = clock_v2.optimize_tsp(coords)
        times.append(time.time() - t0)
        lengths.append(length)
        stats_list.append(stats)
    results['clock_v2'] = {
        'best': min(lengths),
        'mean': np.mean(lengths),
        'time': np.mean(times),
        'gap': 100 * (min(lengths) - optimal) / optimal,
        'stats': stats_list[0]
    }
    
    # Print results
    print(f"\n{'Method':<15} {'Best':>12} {'Gap %':>10} {'Time':>10}")
    print("-" * 50)
    
    for method in ['original', 'clock_v1', 'clock_v2']:
        r = results[method]
        print(f"{method:<15} {r['best']:>12.1f} {r['gap']:>10.2f}% {r['time']:>10.4f}s")
    
    # v2 diagnostics
    if 'stats' in results['clock_v2']:
        s = results['clock_v2']['stats']
        print(f"\nv2 diagnostics: dim={s.instance_dimension:.3f}, "
              f"clusters={s.n_clusters}, resonance={s.resonance_strength:.4f}")
    
    return results


def main():
    print("=" * 70)
    print("TSPLIB BENCHMARK: Clock-Resonant Optimizer v2")
    print("=" * 70)
    
    all_results = {}
    
    for name in ['eil51', 'berlin52', 'st70', 'eil76', 'kroA100']:
        all_results[name] = benchmark_instance(name, n_trials=3)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Gap to Optimal (%)")
    print("=" * 70)
    
    print(f"\n{'Instance':<12} {'Optimal':>10} {'Original':>12} {'Clock v1':>12} {'Clock v2':>12} {'v2 Improv':>12}")
    print("-" * 75)
    
    total_orig = 0
    total_v2 = 0
    
    for name, results in all_results.items():
        optimal = TSPLIB_INSTANCES[name]['optimal']
        orig_gap = results['original']['gap']
        v1_gap = results['clock_v1']['gap']
        v2_gap = results['clock_v2']['gap']
        improvement = 100 * (orig_gap - v2_gap) / orig_gap if orig_gap > 0 else 0
        
        total_orig += orig_gap
        total_v2 += v2_gap
        
        print(f"{name:<12} {optimal:>10} {orig_gap:>12.2f}% {v1_gap:>12.2f}% {v2_gap:>12.2f}% {improvement:>11.1f}%")
    
    avg_orig = total_orig / len(all_results)
    avg_v2 = total_v2 / len(all_results)
    avg_improvement = 100 * (avg_orig - avg_v2) / avg_orig
    
    print("-" * 75)
    print(f"{'AVERAGE':<12} {'':<10} {avg_orig:>12.2f}% {'':<12} {avg_v2:>12.2f}% {avg_improvement:>11.1f}%")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
