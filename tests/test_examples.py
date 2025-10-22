#!/usr/bin/env python3
"""
Test Examples and Notebooks
============================

Automated testing for all example scripts and Jupyter notebooks.
Ensures all examples run without errors.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_python_examples():
    """Test all Python example scripts."""
    print("\n" + "="*70)
    print("TESTING PYTHON EXAMPLES")
    print("="*70)
    
    examples_dir = Path(__file__).parent.parent / "examples"
    example_files = sorted(examples_dir.glob("*.py"))
    
    results = []
    for example_file in example_files:
        print(f"\nTesting {example_file.name}...", end=" ")
        try:
            result = subprocess.run(
                [sys.executable, str(example_file)],
                cwd=str(examples_dir.parent),
                capture_output=True,
                timeout=30,
                text=True
            )
            if result.returncode == 0:
                print("✓ PASS")
                results.append((example_file.name, True, None))
            else:
                print("✗ FAIL")
                error = result.stderr if result.stderr else result.stdout
                results.append((example_file.name, False, error))
        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT")
            results.append((example_file.name, False, "Timeout after 30s"))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((example_file.name, False, str(e)))
    
    return results


def test_notebooks():
    """Test all Jupyter notebooks by converting to Python and executing."""
    print("\n" + "="*70)
    print("TESTING JUPYTER NOTEBOOKS")
    print("="*70)
    
    notebooks_dir = Path(__file__).parent.parent / "examples" / "notebooks"
    notebook_files = sorted(notebooks_dir.glob("*.ipynb"))
    
    # Check if nbconvert is available
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        has_nbconvert = True
    except ImportError:
        print("\n⚠ nbconvert not installed - skipping notebook tests")
        print("  Install with: pip install nbconvert")
        return []
    
    results = []
    for notebook_file in notebook_files:
        print(f"\nTesting {notebook_file.name}...", end=" ")
        try:
            # Read notebook
            with open(notebook_file, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Execute notebook
            ep = ExecutePreprocessor(timeout=180, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': str(notebooks_dir)}})
            
            print("✓ PASS")
            results.append((notebook_file.name, True, None))
            
        except Exception as e:
            print(f"✗ FAIL")
            error_msg = str(e)[:200]  # Truncate long errors
            results.append((notebook_file.name, False, error_msg))
    
    return results


def print_summary(python_results, notebook_results):
    """Print test summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Python examples
    print("\nPython Examples:")
    python_passed = sum(1 for _, passed, _ in python_results if passed)
    python_total = len(python_results)
    
    for name, passed, error in python_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} {name}")
        if error and not passed:
            # Print first line of error
            first_line = error.split('\n')[0][:60]
            print(f"           {first_line}...")
    
    print(f"\n  Results: {python_passed}/{python_total} passed")
    
    # Notebooks
    if notebook_results:
        print("\nJupyter Notebooks:")
        notebook_passed = sum(1 for _, passed, _ in notebook_results if passed)
        notebook_total = len(notebook_results)
        
        for name, passed, error in notebook_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status:8} {name}")
            if error and not passed:
                first_line = error.split('\n')[0][:60]
                print(f"           {first_line}...")
        
        print(f"\n  Results: {notebook_passed}/{notebook_total} passed")
    
    # Overall
    print("\n" + "="*70)
    total_passed = python_passed + (sum(1 for _, p, _ in notebook_results if p) if notebook_results else 0)
    total_tests = python_total + (len(notebook_results) if notebook_results else 0)
    
    if total_passed == total_tests:
        print(f"✓ ALL TESTS PASSED ({total_passed}/{total_tests})")
    else:
        print(f"✗ SOME TESTS FAILED ({total_passed}/{total_tests} passed)")
    
    print("="*70)
    
    return total_passed == total_tests


def main():
    """Run all example tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test examples and notebooks')
    parser.add_argument('--skip-notebooks', action='store_true',
                       help='Skip notebook tests (faster)')
    parser.add_argument('--notebooks-only', action='store_true',
                       help='Only test notebooks')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HOLOGRAPHER'S WORKBENCH - EXAMPLE TESTS")
    print("="*70)
    
    # Test Python examples
    if not args.notebooks_only:
        python_results = test_python_examples()
    else:
        python_results = []
    
    # Test notebooks
    if not args.skip_notebooks:
        notebook_results = test_notebooks()
    else:
        print("\n⚠ Skipping notebook tests (use --skip-notebooks=false to enable)")
        notebook_results = []
    
    # Print summary
    all_passed = print_summary(python_results, notebook_results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
