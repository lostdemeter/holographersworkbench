"""
Example: Using the Formula Code Generator
==========================================

Demonstrates automatic production code generation from formula discoveries.
"""

import numpy as np
from formula_code_generator import FormulaCodeGenerator
from error_pattern_visualizer import ErrorPatternAnalyzer


# Example 1: Generate from Error Analysis
print("=" * 70)
print("EXAMPLE 1: Generate from Error Analysis")
print("=" * 70)

# Analyze errors
x = np.linspace(0, 10, 100)
actual = np.sin(x) + 0.1 * x
predicted = np.sin(x)

analyzer = ErrorPatternAnalyzer(actual, predicted, x, name="Sin + Linear")
report = analyzer.analyze_all()

# Create generator
generator = FormulaCodeGenerator(
    base_formula="np.sin(x)",
    name="improved_sin",
    description="Sine function with linear correction"
)

# Add corrections from analysis
for suggestion in report.suggestions[:2]:
    generator.add_correction(suggestion)

# Generate function
code = generator.generate_function()
print("\nGenerated function:")
print(code)


# Example 2: Generate Complete Module
print("\n" + "=" * 70)
print("EXAMPLE 2: Generate Complete Module")
print("=" * 70)

generator2 = FormulaCodeGenerator(
    base_formula="x**2 + 2*x",
    name="polynomial_formula",
    description="Polynomial with corrections"
)

generator2.add_parameter("coeff1", 0.5, "First coefficient", optimized=True)
generator2.add_correction("correction = coeff1 * np.log(x + 1)")

module_code = generator2.generate_module(
    include_tests=True,
    include_benchmarks=True
)

print(f"\nGenerated module ({len(module_code)} chars)")
print("Preview (first 30 lines):")
print('\n'.join(module_code.split('\n')[:30]))


# Example 3: Validate Generated Code
print("\n" + "=" * 70)
print("EXAMPLE 3: Validate Generated Code")
print("=" * 70)

code = generator.generate_function()
validation = generator.validate_code(code)
validation.print_summary()


# Example 4: Export to File
print("\n" + "=" * 70)
print("EXAMPLE 4: Export to File")
print("=" * 70)

try:
    generator.export_to_file(
        "_staging_formula_gen/improved_formula.py",
        format="module",
        overwrite=True
    )
except Exception as e:
    print(f"Export: {e}")


print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
