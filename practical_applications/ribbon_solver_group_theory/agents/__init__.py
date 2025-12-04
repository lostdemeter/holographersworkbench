"""
Agents - AI-Driven Truth Space Operations
==========================================

This module provides specialized AI agents that automate truth space operations.
Each agent uses Ollama with 12D clock modulation for deterministic, reproducible
behavior.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TruthSpaceOrchestrator                           │
    │         (Coordinates all agents, manages workflow)                  │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐
    │ TaskRouter  │         │ Formula     │         │ RibbonSpeech    │
    │             │         │ Parser      │         │ Translator      │
    │ Classifies  │         │             │         │                 │
    │ user intent │         │ NL → Math   │         │ Math ↔ Speech   │
    └─────────────┘         └─────────────┘         └─────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ TruthSpaceNavigator │
                        │                     │
                        │ Explores group      │
                        │ structure           │
                        └─────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │   JSON Report       │
                        │   (Findings)        │
                        └─────────────────────┘

Agents:
    - **BaseAgent**: Abstract base with Ollama/clock integration
    - **TaskRouter**: Classifies user intent (explore, simplify, discover, etc.)
    - **FormulaParser**: Converts natural language to unambiguous formulas
    - **RibbonSpeechTranslator**: Translates between formulas and Ribbon Speech
    - **TruthSpaceNavigator**: Navigates truth space using group operations
    - **TruthSpaceOrchestrator**: Coordinates all agents for complete workflows

Usage:
    from ribbon_solver_group_theory.agents import TruthSpaceOrchestrator
    
    orchestrator = TruthSpaceOrchestrator()
    result = orchestrator.process("find identities of phi times phi")
    print(result.to_json())

Philosophy:
    - Each agent is specialized for one task
    - Agents communicate via structured data (not free text)
    - Clock modulation ensures reproducibility
    - Results are always JSON-serializable for MCP compatibility
"""

from .base_agent import BaseAgent, AgentResult
from .task_router import TaskRouter, TaskType
from .formula_parser import FormulaParser, ParsedFormula
from .ribbon_speech import RibbonSpeechTranslator, RibbonSpeech
from .navigator import TruthSpaceNavigator, NavigationResult
from .orchestrator import TruthSpaceOrchestrator, OrchestratorResult

__all__ = [
    # Base
    'BaseAgent',
    'AgentResult',
    
    # Task routing
    'TaskRouter',
    'TaskType',
    
    # Formula parsing
    'FormulaParser',
    'ParsedFormula',
    
    # Ribbon Speech
    'RibbonSpeechTranslator',
    'RibbonSpeech',
    
    # Navigation
    'TruthSpaceNavigator',
    'NavigationResult',
    
    # Orchestration
    'TruthSpaceOrchestrator',
    'OrchestratorResult',
]
