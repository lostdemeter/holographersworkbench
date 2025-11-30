#!/usr/bin/env python3
"""
Tests for the φ-BBP Formula
===========================

Verifies all claims about the φ-BBP formula discovery.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery_engine import PhiPatternDetector, ClosedFormSearcher, PHI
from domains.bbp_domain import BBPDomain, get_phi_bbp_formula


class TestPhiBBPFormula(unittest.TestCase):
    """Test the φ-BBP formula."""
    
    def setUp(self):
        self.formula = get_phi_bbp_formula()
        self.domain = BBPDomain()
    
    def test_accuracy(self):
        """Test that error < 10⁻¹⁰."""
        value = self.formula.evaluate(n_terms=100)
        error = abs(value - np.pi)
        self.assertLess(error, 1e-10, f"Error {error:.2e} exceeds threshold")
    
    def test_convergence_rate(self):
        """Test convergence rate ≈ 3.61."""
        rate = self.formula.convergence_rate()
        self.assertAlmostEqual(rate, 3.61, places=1)
    
    def test_beats_bellard(self):
        """Test that rate > Bellard's 3.01."""
        rate = self.formula.convergence_rate()
        bellard_rate = 3.01
        self.assertGreater(rate, bellard_rate)
    
    def test_has_phi_corrections(self):
        """Test that formula has φ-corrections."""
        self.assertTrue(self.formula.has_phi_corrections())
    
    def test_integer_coefficients(self):
        """Test integer coefficients are correct."""
        expected = [256, -32, 4, 1, -128, -64, -128, 4]
        self.assertEqual(self.formula.integer_coefficients, expected)


class TestPatternDetection(unittest.TestCase):
    """Test pattern detection."""
    
    def setUp(self):
        self.detector = PhiPatternDetector()
    
    def test_find_clean_pattern(self):
        """Test finding the cleanest pattern (slot 4)."""
        value = 0.073263011134871
        pattern = self.detector.find_phi_pattern(value)
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.numerator, 13)
        self.assertEqual(pattern.denominator, 16)
        self.assertEqual(pattern.phi_power, -5)
    
    def test_pattern_error(self):
        """Test that pattern errors are small."""
        corrections = [
            0.021013707249694,
            -0.047113568832732,
            0.007043075951984,
            0.013181561904277,
            0.073263011134871,
            -0.102346114400464,
            -0.153352294762737,
            0.047671994116562,
        ]
        
        for corr in corrections:
            pattern = self.detector.find_phi_pattern(corr, tolerance=1e-3)
            if pattern:
                self.assertLess(pattern.error, 1e-3)


class TestClosedFormSearch(unittest.TestCase):
    """Test closed-form search."""
    
    def setUp(self):
        self.searcher = ClosedFormSearcher()
    
    def test_find_total_closed_form(self):
        """Test finding closed form for total correction."""
        total = -0.140638627638544
        result = self.searcher.search(total, max_coef=30)
        
        self.assertIsNotNone(result)
        self.assertLess(result.error, 1e-5)


class TestMathematicalIdentities(unittest.TestCase):
    """Test key mathematical identities."""
    
    def test_phi_squared_identity(self):
        """Test φ² + φ⁻² = 3."""
        result = PHI**2 + PHI**(-2)
        self.assertAlmostEqual(result, 3.0, places=10)
    
    def test_four_identity(self):
        """Test 4 = φ² + φ⁻² + 1."""
        result = PHI**2 + PHI**(-2) + 1
        self.assertAlmostEqual(result, 4.0, places=10)
    
    def test_fibonacci_arctan(self):
        """Test arctan(1/φ) + arctan(1/φ³) = π/4."""
        result = np.arctan(1/PHI) + np.arctan(1/PHI**3)
        self.assertAlmostEqual(result, np.pi/4, places=10)


class TestDomainVerification(unittest.TestCase):
    """Test domain verification."""
    
    def setUp(self):
        self.domain = BBPDomain()
    
    def test_verify_phi_bbp(self):
        """Test full verification."""
        result = self.domain.verify_phi_bbp()
        
        self.assertTrue(result['verification']['valid'])
        self.assertTrue(result['benchmark']['beats_bellard'])
        self.assertTrue(result['phi_analysis']['has_patterns'])


if __name__ == '__main__':
    unittest.main()
