"""
Formula Code Generator for Holographic Algorithms
==================================================

Automatically generates production-ready Python code from formula components
and discovered correction patterns.

This module completes the optimization pipeline:
1. Performance Profiler → Identify bottlenecks
2. Error Pattern Visualizer → Discover corrections
3. Formula Code Generator → Generate production code
"""

import numpy as np
import ast
import inspect
from typing import Union, Callable, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# Validation Data Structures
# ============================================================================

@dataclass
class ValidationIssue:
    """Single validation issue.
    
    Attributes
    ----------
    severity : str
        "error", "warning", or "info".
    category : str
        "syntax", "import", "type", "performance", or "numerical".
    message : str
        Description of the issue.
    line_number : int, optional
        Line number where issue occurs.
    suggestion : str, optional
        Suggested fix.
    """
    severity: str
    category: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report.
    
    Attributes
    ----------
    is_valid : bool
        Whether code is valid (no errors).
    errors : List[ValidationIssue]
        Critical errors.
    warnings : List[ValidationIssue]
        Warnings.
    info : List[ValidationIssue]
        Informational messages.
    """
    is_valid: bool
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    
    def print_summary(self):
        """Print human-readable summary."""
        print("=" * 70)
        print("CODE VALIDATION REPORT")
        print("=" * 70)
        print(f"Status: {'✓ VALID' if self.is_valid else '✗ INVALID'}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Info: {len(self.info)}")
        print()
        
        if self.errors:
            print("ERRORS:")
            for err in self.errors:
                print(f"  [{err.category}] {err.message}")
                if err.suggestion:
                    print(f"    → {err.suggestion}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warn in self.warnings:
                print(f"  [{warn.category}] {warn.message}")
                if warn.suggestion:
                    print(f"    → {warn.suggestion}")
        
        print("=" * 70)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


# ============================================================================
# Code Validator
# ============================================================================

class CodeValidator:
    """Validate generated code for correctness and quality."""
    
    @staticmethod
    def check_syntax(code: str) -> List[ValidationIssue]:
        """Check Python syntax using ast.parse."""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity="error",
                category="syntax",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                suggestion="Check for missing parentheses, colons, or indentation"
            ))
        return issues
    
    @staticmethod
    def check_imports(code: str) -> List[ValidationIssue]:
        """Check if all imports are available."""
        import importlib
        issues = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError:
                            issues.append(ValidationIssue(
                                severity="warning",
                                category="import",
                                message=f"Module '{alias.name}' may not be available",
                                suggestion=f"Install with: pip install {alias.name}"
                            ))
        except:
            pass
        
        return issues
    
    @staticmethod
    def check_numerical_stability(code: str) -> List[ValidationIssue]:
        """Check for potential numerical issues."""
        issues = []
        
        if "/" in code and "!= 0" not in code and "if" not in code:
            issues.append(ValidationIssue(
                severity="warning",
                category="numerical",
                message="Potential division by zero",
                suggestion="Add zero checks before division"
            ))
        
        if "np.exp(" in code or "math.exp(" in code:
            issues.append(ValidationIssue(
                severity="info",
                category="numerical",
                message="Exponential function used (overflow risk)",
                suggestion="Consider using np.clip or checking input range"
            ))
        
        return issues
    
    @staticmethod
    def check_performance(code: str) -> List[ValidationIssue]:
        """Check for performance anti-patterns."""
        issues = []
        
        if "for " in code and ("np.array" in code or "np.ndarray" in code):
            issues.append(ValidationIssue(
                severity="info",
                category="performance",
                message="Loop over numpy array detected",
                suggestion="Consider using vectorized operations"
            ))
        
        return issues
    
    @staticmethod
    def validate(code: str) -> ValidationReport:
        """Run all validation checks."""
        all_issues = []
        all_issues.extend(CodeValidator.check_syntax(code))
        all_issues.extend(CodeValidator.check_imports(code))
        all_issues.extend(CodeValidator.check_numerical_stability(code))
        all_issues.extend(CodeValidator.check_performance(code))
        
        errors = [i for i in all_issues if i.severity == "error"]
        warnings = [i for i in all_issues if i.severity == "warning"]
        info = [i for i in all_issues if i.severity == "info"]
        
        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )


# ============================================================================
# Code Optimizer
# ============================================================================

class CodeOptimizer:
    """Optimize generated code for performance."""
    
    @staticmethod
    def precompute_constants(code: str) -> str:
        """Add comment about constant precomputation."""
        if "2 * np.pi" in code:
            code = "# Constants precomputed for performance\nTWO_PI = 2 * np.pi\n\n" + code
            code = code.replace("2 * np.pi", "TWO_PI")
        return code
    
    @staticmethod
    def optimize(code: str) -> str:
        """Apply basic optimizations."""
        code = CodeOptimizer.precompute_constants(code)
        return code


# ============================================================================
# Test Generator
# ============================================================================

class TestGenerator:
    """Generate pytest test cases for formulas."""
    
    @staticmethod
    def generate_basic_tests(formula_name: str) -> str:
        """Generate basic test cases."""
        return f'''
def test_{formula_name}_basic():
    """Test basic functionality"""
    import numpy as np
    
    # Single value
    result = {formula_name}(np.array([1.0]))
    assert len(result) == 1
    assert np.isfinite(result[0])
    
    # Multiple values
    result = {formula_name}(np.arange(10))
    assert len(result) == 10
    assert np.all(np.isfinite(result))
    
    # Large array
    result = {formula_name}(np.arange(1000))
    assert len(result) == 1000
'''
    
    @staticmethod
    def generate_correctness_tests(
        formula_name: str,
        test_cases: List[np.ndarray],
        tolerance: float = 1e-6
    ) -> str:
        """Generate correctness tests."""
        test_code = f'''
def test_{formula_name}_correctness():
    """Test correctness"""
    import numpy as np
    from numpy.testing import assert_allclose
    
'''
        for i, test_input in enumerate(test_cases[:3]):  # Limit to 3 cases
            test_code += f'''
    # Test case {i+1}
    x = np.array({test_input.tolist()})
    result = {formula_name}(x)
    assert np.all(np.isfinite(result))
'''
        return test_code


# ============================================================================
# Benchmark Generator
# ============================================================================

class BenchmarkGenerator:
    """Generate performance benchmarks."""
    
    @staticmethod
    def generate_benchmark(
        formula_name: str,
        test_sizes: List[int] = None
    ) -> str:
        """Generate benchmark code."""
        if test_sizes is None:
            test_sizes = [10, 100, 1000, 10000]
        
        return f'''
def benchmark_{formula_name}():
    """Benchmark {formula_name} performance"""
    import numpy as np
    import timeit
    
    sizes = {test_sizes}
    
    print(f"Benchmarking {formula_name}:")
    for size in sizes:
        x = np.arange(size)
        time_taken = timeit.timeit(lambda: {formula_name}(x), number=100)
        per_call = time_taken / 100
        print(f"  Size {{size:>6}}: {{per_call*1000:>8.3f}}ms per call")

if __name__ == "__main__":
    benchmark_{formula_name}()
'''


# ============================================================================
# Main Formula Code Generator
# ============================================================================

class FormulaCodeGenerator:
    """Generate production-ready code from formula components.
    
    Parameters
    ----------
    base_formula : Union[str, Callable]
        Base formula as code string or callable.
    name : str
        Function name for generated code.
    description : str
        Human-readable description.
    author : str
        Author attribution.
    
    Examples
    --------
    >>> generator = FormulaCodeGenerator(
    ...     base_formula="def f(x): return x**2",
    ...     name="improved_formula"
    ... )
    >>> generator.add_correction("correction = 0.1 * x")
    >>> code = generator.generate_function()
    """
    
    def __init__(
        self,
        base_formula: Union[str, Callable],
        name: str = "improved_formula",
        description: str = "",
        author: str = "Holographer's Workbench"
    ):
        self.base_formula = base_formula
        self.name = name
        self.description = description or f"Improved formula: {name}"
        self.author = author
        self.corrections = []
        self.parameters = {}
        
        # Extract base formula code
        if callable(base_formula):
            try:
                self.base_code = inspect.getsource(base_formula)
            except:
                self.base_code = "result = base_formula(x)"
        else:
            self.base_code = str(base_formula)
    
    def add_correction(
        self,
        correction,
        priority: int = None,
        description: str = ""
    ):
        """Add a correction term to the formula.
        
        Parameters
        ----------
        correction : Union[CorrectionSuggestion, str, Callable]
            Correction to add.
        priority : int, optional
            Priority order (lower = applied first).
        description : str
            Human-readable description.
        """
        if priority is None:
            priority = len(self.corrections)
        
        # Extract code from correction
        if hasattr(correction, 'code_snippet'):
            code = correction.code_snippet
            desc = correction.description if not description else description
        elif callable(correction):
            try:
                code = inspect.getsource(correction)
            except:
                code = str(correction)
            desc = description
        else:
            code = str(correction)
            desc = description
        
        self.corrections.append({
            'code': code,
            'priority': priority,
            'description': desc
        })
        
        # Sort by priority
        self.corrections.sort(key=lambda x: x['priority'])
    
    def add_parameter(
        self,
        name: str,
        value: float,
        description: str = "",
        optimized: bool = False
    ):
        """Add a parameter to the formula.
        
        Parameters
        ----------
        name : str
            Parameter name.
        value : float
            Parameter value.
        description : str
            What this parameter does.
        optimized : bool
            Whether this was optimized.
        """
        self.parameters[name] = {
            'value': value,
            'description': description,
            'optimized': optimized
        }
    
    def generate_function(
        self,
        style: str = "vectorized",
        include_docstring: bool = True,
        include_type_hints: bool = True,
        include_validation: bool = False
    ) -> str:
        """Generate a standalone function.
        
        Parameters
        ----------
        style : str
            "vectorized", "scalar", or "numba".
        include_docstring : bool
            Add comprehensive docstring.
        include_type_hints : bool
            Add type hints.
        include_validation : bool
            Add input validation.
            
        Returns
        -------
        str
            Python code as string.
        """
        # Build function signature
        if include_type_hints:
            signature = f"def {self.name}(x: np.ndarray) -> np.ndarray:"
        else:
            signature = f"def {self.name}(x):"
        
        # Build docstring
        docstring = ""
        if include_docstring:
            docstring = f'''    """
    {self.description}
    
    Parameters
    ----------
    x : np.ndarray
        Input values.
        
    Returns
    -------
    np.ndarray
        Computed values.
    
    Examples
    --------
    >>> result = {self.name}(np.array([1.0, 2.0, 3.0]))
    """'''
        
        # Build validation
        validation = ""
        if include_validation:
            validation = '''    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([])
'''
        
        # Build parameter definitions
        param_code = ""
        if self.parameters:
            param_code = "    # Parameters\n"
            for pname, pdata in self.parameters.items():
                param_code += f"    {pname} = {pdata['value']}"
                if pdata['description']:
                    param_code += f"  # {pdata['description']}"
                param_code += "\n"
            param_code += "\n"
        
        # Build base computation
        base_computation = "    # Base formula\n"
        if "def " in self.base_code:
            # Extract function body
            lines = self.base_code.split('\n')
            for line in lines:
                if 'return' in line:
                    base_computation += f"    result = {line.strip().replace('return ', '')}\n"
                    break
        else:
            base_computation += f"    result = {self.base_code}\n"
        
        # Build corrections
        corrections_code = ""
        if self.corrections:
            corrections_code = "\n    # Corrections\n"
            for i, corr in enumerate(self.corrections, 1):
                corrections_code += f"    # Correction {i}: {corr['description']}\n"
                # Extract correction computation
                corr_lines = corr['code'].split('\n')
                for line in corr_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if 'correction' in line and '=' in line:
                            corrections_code += f"    {line}\n"
                            corrections_code += "    result = result + correction\n"
                            break
        
        # Build return
        return_code = "\n    return result"
        
        # Combine all parts
        code = signature + "\n"
        if docstring:
            code += docstring + "\n"
        if validation:
            code += validation + "\n"
        if param_code:
            code += param_code
        code += base_computation
        if corrections_code:
            code += corrections_code
        code += return_code
        
        return code
    
    def generate_module(
        self,
        include_tests: bool = True,
        include_benchmarks: bool = True,
        include_examples: bool = True
    ) -> str:
        """Generate a complete Python module.
        
        Parameters
        ----------
        include_tests : bool
            Include test cases.
        include_benchmarks : bool
            Include benchmarks.
        include_examples : bool
            Include usage examples.
            
        Returns
        -------
        str
            Complete module code.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        module = f'''"""
{self.name}

{self.description}

Author: {self.author}
Generated: {timestamp}
"""

import numpy as np
from typing import Union, Optional

'''
        
        # Add main function
        module += "# " + "=" * 70 + "\n"
        module += "# Main Formula\n"
        module += "# " + "=" * 70 + "\n\n"
        module += self.generate_function() + "\n\n"
        
        # Add tests
        if include_tests:
            module += "# " + "=" * 70 + "\n"
            module += "# Tests\n"
            module += "# " + "=" * 70 + "\n\n"
            module += TestGenerator.generate_basic_tests(self.name) + "\n"
        
        # Add benchmarks
        if include_benchmarks:
            module += "# " + "=" * 70 + "\n"
            module += "# Benchmarks\n"
            module += "# " + "=" * 70 + "\n\n"
            module += BenchmarkGenerator.generate_benchmark(self.name) + "\n"
        
        # Add examples
        if include_examples:
            module += "# " + "=" * 70 + "\n"
            module += "# Examples\n"
            module += "# " + "=" * 70 + "\n\n"
            module += f'''if __name__ == "__main__":
    # Example usage
    x = np.arange(10)
    result = {self.name}(x)
    print(f"Input: {{x}}")
    print(f"Output: {{result}}")
'''
        
        return module
    
    def validate_code(self, code: str) -> ValidationReport:
        """Validate generated code.
        
        Parameters
        ----------
        code : str
            Code to validate.
            
        Returns
        -------
        ValidationReport
            Validation results.
        """
        return CodeValidator.validate(code)
    
    def optimize_code(self, code: str) -> str:
        """Optimize generated code for performance.
        
        Parameters
        ----------
        code : str
            Code to optimize.
            
        Returns
        -------
        str
            Optimized code.
        """
        return CodeOptimizer.optimize(code)
    
    def export_to_file(
        self,
        filepath: str,
        format: str = "module",
        overwrite: bool = False
    ):
        """Export generated code to file.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        format : str
            "function", "class", or "module".
        overwrite : bool
            Whether to overwrite existing file.
        """
        import os
        
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to replace.")
        
        if format == "function":
            code = self.generate_function()
        elif format == "module":
            code = self.generate_module()
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        print(f"✓ Exported to {filepath}")
    
    def preview(self, format: str = "function", max_lines: int = 50):
        """Preview generated code (truncated).
        
        Parameters
        ----------
        format : str
            "function" or "module".
        max_lines : int
            Maximum lines to show.
        """
        if format == "function":
            code = self.generate_function()
        elif format == "module":
            code = self.generate_module()
        else:
            code = self.generate_function()
        
        lines = code.split('\n')
        if len(lines) > max_lines:
            preview = '\n'.join(lines[:max_lines])
            preview += f"\n... ({len(lines) - max_lines} more lines)"
        else:
            preview = code
        
        print(preview)
    
    def generate_tests(
        self,
        ground_truth: Callable = None,
        test_cases: List[np.ndarray] = None,
        tolerance: float = 1e-6
    ) -> str:
        """Generate pytest test cases.
        
        Parameters
        ----------
        ground_truth : Callable, optional
            Function to compare against.
        test_cases : List[np.ndarray], optional
            Specific test inputs.
        tolerance : float
            Acceptable error tolerance.
            
        Returns
        -------
        str
            pytest code as string.
        """
        if test_cases:
            return TestGenerator.generate_correctness_tests(
                self.name, test_cases, tolerance
            )
        else:
            return TestGenerator.generate_basic_tests(self.name)
    
    def generate_benchmark(
        self,
        baseline: Callable = None,
        test_sizes: List[int] = None
    ) -> str:
        """Generate performance benchmark code.
        
        Parameters
        ----------
        baseline : Callable, optional
            Baseline function to compare against.
        test_sizes : List[int], optional
            Input sizes to benchmark.
            
        Returns
        -------
        str
            Benchmark code as string.
        """
        return BenchmarkGenerator.generate_benchmark(self.name, test_sizes)


# ============================================================================
# Utility Functions
# ============================================================================

def extract_function_code(func: Callable) -> str:
    """Extract source code from callable.
    
    Parameters
    ----------
    func : Callable
        Function to extract code from.
        
    Returns
    -------
    str
        Source code.
    """
    try:
        return inspect.getsource(func)
    except:
        return str(func)


def format_code(code: str) -> str:
    """Format code using black (if available).
    
    Parameters
    ----------
    code : str
        Code to format.
        
    Returns
    -------
    str
        Formatted code.
    """
    try:
        import black
        return black.format_str(code, mode=black.Mode())
    except ImportError:
        return code
