"""
Truth Space Orchestrator
========================

Coordinates all agents to process user queries end-to-end.

Pipeline:
1. TaskRouter: Classify user intent
2. FormulaParser: Extract and normalize formula
3. RibbonSpeechTranslator: Convert to/from Ribbon Speech
4. TruthSpaceNavigator: Execute the task
5. Report: Generate JSON findings

This is the main entry point for AI-driven truth space operations.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import time

from .base_agent import BaseAgent, AgentResult
from .task_router import TaskRouter, TaskType
from .formula_parser import FormulaParser
from .ribbon_speech import RibbonSpeechTranslator
from .navigator import TruthSpaceNavigator


@dataclass
class OrchestratorResult:
    """Complete result from orchestrated processing."""
    success: bool
    query: str
    task_type: str
    
    # Pipeline stages
    parsed_formula: Optional[str]
    ribbon_speech: Optional[str]
    
    # Findings
    findings: List[Dict[str, Any]]
    summary: str
    
    # Metadata
    timestamp: str
    total_time_ms: float
    agent_times: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def print_report(self):
        """Print a human-readable report."""
        print("=" * 70)
        print("TRUTH SPACE ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nQuery: {self.query}")
        print(f"Task Type: {self.task_type}")
        print(f"Status: {'✓ Success' if self.success else '✗ Failed'}")
        
        if self.parsed_formula:
            print(f"\nParsed Formula: {self.parsed_formula}")
        
        if self.ribbon_speech:
            print(f"Ribbon Speech: {self.ribbon_speech}")
        
        print(f"\n{'-' * 70}")
        print("FINDINGS")
        print(f"{'-' * 70}")
        
        if self.findings:
            for i, finding in enumerate(self.findings, 1):
                print(f"\n{i}. {finding.get('title', 'Finding')}")
                for key, value in finding.items():
                    if key != 'title':
                        print(f"   {key}: {value}")
        else:
            print("No specific findings.")
        
        print(f"\n{'-' * 70}")
        print("SUMMARY")
        print(f"{'-' * 70}")
        print(self.summary)
        
        print(f"\n{'-' * 70}")
        print(f"Total time: {self.total_time_ms:.1f}ms")
        print("=" * 70)


class TruthSpaceOrchestrator(BaseAgent):
    """
    Orchestrates all agents for complete truth space operations.
    
    Usage:
        orchestrator = TruthSpaceOrchestrator()
        result = orchestrator.process("find identities of phi times phi")
        result.print_report()
        print(result.to_json())
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize all agents
        agent_kwargs = {
            'model': self.model,
            'clock_position': self.clock_position,
            'verbose': False,  # Agents are quiet, orchestrator reports
            'use_llm': self.use_llm,
        }
        
        self.router = TaskRouter(**agent_kwargs)
        self.parser = FormulaParser(**agent_kwargs)
        self.translator = RibbonSpeechTranslator(**agent_kwargs)
        self.navigator = TruthSpaceNavigator(**agent_kwargs)
        
        self.agent_times = {}
    
    def process(self, query: str) -> OrchestratorResult:
        """
        Process a user query through the complete pipeline.
        
        Args:
            query: Natural language query
            
        Returns:
            OrchestratorResult with complete analysis
        """
        self._start_timer()
        self._log(f"Processing: {query}")
        
        findings = []
        parsed_formula = None
        ribbon_speech = None
        summary = ""
        
        try:
            # Stage 1: Route the task
            self._log("Stage 1: Routing task...")
            route_start = time.perf_counter()
            route_result = self.router.process(query)
            self.agent_times['router'] = (time.perf_counter() - route_start) * 1000
            
            task_type = TaskType[route_result.data['task_type']]
            extracted_formula = route_result.data.get('extracted_formula')
            
            self._log(f"  → Task type: {task_type.name}")
            
            # Stage 2: Parse formula
            self._log("Stage 2: Parsing formula...")
            parse_start = time.perf_counter()
            
            if extracted_formula:
                parse_result = self.parser.process(extracted_formula)
                parsed_formula = parse_result.data.get('explicit', extracted_formula)
            else:
                # Try to extract from query directly
                parse_result = self.parser.process(query)
                parsed_formula = parse_result.data.get('explicit')
            
            self.agent_times['parser'] = (time.perf_counter() - parse_start) * 1000
            self._log(f"  → Parsed: {parsed_formula}")
            
            # Stage 3: Translate to Ribbon Speech
            self._log("Stage 3: Translating to Ribbon Speech...")
            translate_start = time.perf_counter()
            
            if parsed_formula:
                translate_result = self.translator.process(parsed_formula)
                ribbon_speech = translate_result.data.get('speech')
            
            self.agent_times['translator'] = (time.perf_counter() - translate_start) * 1000
            self._log(f"  → Ribbon Speech: {ribbon_speech}")
            
            # Stage 4: Execute task
            self._log(f"Stage 4: Executing {task_type.name} task...")
            execute_start = time.perf_counter()
            
            if task_type == TaskType.EXPLORE:
                findings, summary = self._execute_explore(parsed_formula)
            elif task_type == TaskType.SIMPLIFY:
                findings, summary = self._execute_simplify(parsed_formula)
            elif task_type == TaskType.DISCOVER:
                findings, summary = self._execute_discover(parsed_formula)
            elif task_type == TaskType.VERIFY:
                findings, summary = self._execute_verify(parsed_formula)
            elif task_type == TaskType.TRANSLATE:
                findings, summary = self._execute_translate(parsed_formula, ribbon_speech)
            elif task_type == TaskType.ANALYZE:
                findings, summary = self._execute_analyze(parsed_formula)
            else:
                summary = "Could not determine task type"
            
            self.agent_times['executor'] = (time.perf_counter() - execute_start) * 1000
            
            success = len(findings) > 0 or summary
            
        except Exception as e:
            self._log(f"Error: {e}")
            success = False
            summary = f"Error during processing: {e}"
            task_type = TaskType.UNKNOWN
        
        return OrchestratorResult(
            success=success,
            query=query,
            task_type=task_type.name,
            parsed_formula=parsed_formula,
            ribbon_speech=ribbon_speech,
            findings=findings,
            summary=summary,
            timestamp=self._timestamp(),
            total_time_ms=self._elapsed_ms(),
            agent_times=self.agent_times,
        )
    
    def _execute_explore(self, formula: str) -> tuple:
        """Execute an exploration task."""
        if not formula:
            return [], "No formula to explore"
        
        nav_result = self.navigator.process(formula)
        
        findings = []
        
        # Add starting position
        start_data = nav_result.data.get('start', {})
        findings.append({
            'title': 'Starting Position',
            'expression': start_data.get('expression'),
            'category': start_data.get('category'),
            'dominant_anchor': start_data.get('dominant_anchor'),
            'ribbon_speech': start_data.get('ribbon_speech'),
        })
        
        # Add interesting finds
        for find in nav_result.data.get('interesting_finds', []):
            findings.append({
                'title': 'Related Truth',
                'expression': find.get('expression'),
                'operation': find.get('operation'),
                'ribbon_speech': find.get('ribbon_speech'),
                'reason': find.get('reason'),
            })
        
        summary = nav_result.data.get('summary', '')
        
        return findings, summary
    
    def _execute_simplify(self, formula: str) -> tuple:
        """Execute a simplification task."""
        if not formula:
            return [], "No formula to simplify"
        
        # Use navigator to find simplifications
        nav_result = self.navigator.process(formula)
        
        findings = []
        
        # Look for simplifications in interesting finds
        for find in nav_result.data.get('interesting_finds', []):
            if find.get('operation') == 'simplification':
                findings.append({
                    'title': 'Simplification Found',
                    'original': formula,
                    'simplified': find.get('expression'),
                    'reason': find.get('reason'),
                })
        
        if findings:
            summary = f"Found {len(findings)} simplification(s) for '{formula}'"
        else:
            summary = f"No simplifications found for '{formula}'"
        
        return findings, summary
    
    def _execute_discover(self, formula: str) -> tuple:
        """Execute a discovery task."""
        # Import processors for discovery
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from processors import IdentityMiner, SymmetryFinder
        from core import TruthGroup
        
        group = TruthGroup()
        findings = []
        
        if formula:
            # Find symmetries of the formula
            finder = SymmetryFinder(group, verbose=False)
            sym_result = finder.find_symmetries(formula)
            
            for simp in sym_result.simplifications:
                findings.append({
                    'title': 'Identity Discovered',
                    'expression': formula,
                    'equals': simp.get('simplified'),
                    'category': simp.get('category'),
                    'verified': simp.get('verified'),
                })
        else:
            # General identity mining
            miner = IdentityMiner(group, verbose=False)
            result = miner.mine_pythagorean()
            
            for identity in result.all_identities[:5]:
                findings.append({
                    'title': 'Known Identity',
                    'lhs': identity.get('lhs'),
                    'rhs': identity.get('rhs'),
                    'category': identity.get('category'),
                })
        
        summary = f"Discovered {len(findings)} identities"
        return findings, summary
    
    def _execute_verify(self, formula: str) -> tuple:
        """Execute a verification task."""
        if not formula:
            return [], "No formula to verify"
        
        # Check if it's an equation (contains =)
        if '=' in formula:
            parts = formula.split('=')
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                
                # Try numerical verification
                verified = self._verify_numerically(lhs, rhs)
                
                findings = [{
                    'title': 'Verification Result',
                    'lhs': lhs,
                    'rhs': rhs,
                    'verified': verified,
                    'method': 'numerical',
                }]
                
                summary = f"'{formula}' is {'VERIFIED ✓' if verified else 'NOT VERIFIED ✗'}"
                return findings, summary
        
        return [], f"Cannot verify '{formula}' - not an equation"
    
    def _execute_translate(self, formula: str, ribbon_speech: str) -> tuple:
        """Execute a translation task."""
        findings = []
        
        if formula and ribbon_speech:
            findings.append({
                'title': 'Translation',
                'formula': formula,
                'ribbon_speech': ribbon_speech,
            })
            
            # Also get anchor analysis
            translate_result = self.translator.process(formula)
            findings.append({
                'title': 'Anchor Analysis',
                'dominant_anchor': translate_result.data.get('dominant_anchor'),
                'concepts': translate_result.data.get('concepts'),
            })
        
        summary = f"Translated '{formula}' to Ribbon Speech"
        return findings, summary
    
    def _execute_analyze(self, formula: str) -> tuple:
        """Execute an analysis task."""
        if not formula:
            return [], "No formula to analyze"
        
        # Get full analysis
        nav_result = self.navigator.process(formula)
        translate_result = self.translator.process(formula)
        
        findings = [{
            'title': 'Expression Analysis',
            'expression': formula,
            'category': nav_result.data.get('start', {}).get('category'),
            'dominant_anchor': nav_result.data.get('start', {}).get('dominant_anchor'),
            'ribbon_speech': translate_result.data.get('speech'),
            'concepts': translate_result.data.get('concepts'),
            'anchor_weights': translate_result.data.get('anchor_weights'),
        }]
        
        summary = f"Analysis complete for '{formula}'"
        return findings, summary
    
    def _verify_numerically(self, lhs: str, rhs: str) -> bool:
        """Verify two expressions are numerically equal."""
        import numpy as np
        
        context = {
            'x': 0.5, 'y': 0.3, 'z': 0.7,
            'np': np, 'pi': np.pi, 'e': np.e,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'phi': (1 + np.sqrt(5)) / 2,
        }
        
        try:
            # Normalize
            lhs_norm = lhs.replace('²', '**2').replace('·', '*')
            rhs_norm = rhs.replace('²', '**2').replace('·', '*')
            
            lhs_val = eval(lhs_norm, {"__builtins__": {}}, context)
            rhs_val = eval(rhs_norm, {"__builtins__": {}}, context)
            
            return abs(lhs_val - rhs_val) < 1e-10
        except:
            return False


def demo():
    """Demonstrate the orchestrator."""
    print("=" * 70)
    print("TRUTH SPACE ORCHESTRATOR DEMO")
    print("=" * 70)
    
    orchestrator = TruthSpaceOrchestrator(verbose=True)
    
    queries = [
        "find identities of phi times phi",
        "simplify sin squared x plus cos squared x",
        "what is the meaning of the golden ratio?",
        "verify that phi squared equals phi plus one",
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        result = orchestrator.process(query)
        result.print_report()
        print()


if __name__ == "__main__":
    demo()
