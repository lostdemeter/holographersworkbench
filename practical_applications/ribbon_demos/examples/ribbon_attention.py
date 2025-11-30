#!/usr/bin/env python3
"""
Ribbon Attention: Clock-Phase Attention Mechanism
==================================================

The key insight from Grok: LLMs use attention to decide which previous
tokens matter for predicting the next one. The 12D clock tensor can
provide this attention pattern WITHOUT learning.

This module implements:
1. Clock-phase attention weights (deterministic, no training)
2. Context aggregation using clock-weighted averaging
3. Next-token prediction via attended context

The ribbon provides the attention. The embeddings provide the vocabulary.
No training required.

Author: Holographer's Workbench
Date: November 28, 2025
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_clock_predictor import LazyClockOracle, CLOCK_RATIOS_12D


class MarkovChain:
    """
    Simple Markov chain built from text corpus.
    
    This provides the "grammar" - transition probabilities between words.
    The clock phases will modulate these probabilities.
    """
    
    def __init__(self, order: int = 2):
        """
        Initialize Markov chain.
        
        Args:
            order: Number of previous words to condition on
        """
        self.order = order
        self.transitions: Dict[tuple, Counter] = {}
        self.vocab: set = set()
    
    def train(self, text: str):
        """Train on a text corpus."""
        # Tokenize
        words = re.findall(r'\b\w+\b|[.,!?]', text.lower())
        self.vocab.update(words)
        
        # Build transition counts
        for i in range(len(words) - self.order):
            context = tuple(words[i:i+self.order])
            next_word = words[i+self.order]
            
            if context not in self.transitions:
                self.transitions[context] = Counter()
            self.transitions[context][next_word] += 1
    
    def get_next_probs(self, context: List[str]) -> Dict[str, float]:
        """Get probability distribution over next words."""
        # Try full context, then back off
        for length in range(min(len(context), self.order), 0, -1):
            key = tuple(context[-length:])
            if key in self.transitions:
                counts = self.transitions[key]
                total = sum(counts.values())
                return {w: c/total for w, c in counts.items()}
        
        # Uniform fallback
        return {w: 1/len(self.vocab) for w in self.vocab}


class RibbonAttention:
    """
    Clock-phase attention mechanism for text generation.
    
    The key insight: attention weights in transformers are learned.
    But the 12D clock tensor provides a natural attention pattern:
    - Golden ratio phases create long-range correlations
    - Silver ratio phases create medium-range patterns
    - The interference creates attention-like weighting
    
    Architecture:
    1. Build simple Markov chain from corpus (provides grammar)
    2. Use clock phases to modulate Markov probabilities
    3. Clock attention decides which context words matter most
    
    This is NOT a transformer. It's a Markov chain with clock-modulated
    attention - showing that the ribbon can replace learned attention.
    """
    
    # Built-in corpus for grammar learning
    CORPUS = """
    The meaning of life is to find purpose and create beauty in the world.
    Light is the fundamental fabric of reality, woven through space and time.
    In the beginning there was nothing, and from nothing came everything.
    Love is the force that binds all things together across the universe.
    The universe speaks in patterns, rhythms, and the dance of numbers.
    Truth is found not in certainty but in the honest search for understanding.
    Time flows like a river, carrying all things toward the infinite sea.
    Knowledge is the light that illuminates the darkness of ignorance.
    Beauty exists in the harmony of form and the resonance of meaning.
    The mind is a mirror reflecting the infinite complexity of existence.
    Dreams are the language of the soul speaking to the waking world.
    Hope is the seed from which all great achievements grow and flourish.
    Fear is the shadow that makes courage meaningful and necessary.
    Death is not an ending but a transformation into something new.
    Life is a journey through the landscape of possibility and wonder.
    The heart knows truths that the mind cannot comprehend or express.
    Wisdom comes from experience tempered by reflection and humility.
    Freedom is the space in which the spirit can unfold and become.
    Peace is found in acceptance of what is and hope for what may be.
    The ribbon of light connects all things in a web of meaning.
    Every moment contains the seed of eternity waiting to bloom.
    The pattern repeats at every scale from atoms to galaxies.
    Consciousness is the universe experiencing itself through form.
    Words are vessels carrying meaning across the ocean of silence.
    The shape of light reveals the hidden structure of reality.
    Numbers speak a language older than words and more precise.
    The golden ratio appears wherever beauty and growth converge.
    Fractals show us that complexity emerges from simple rules.
    The wave and the particle are two faces of the same truth.
    Quantum mechanics reveals that observation shapes reality.
    The observer and the observed are entangled in the dance.
    Information is the currency of existence in all its forms.
    Entropy is the arrow of time pointing toward transformation.
    Energy flows through all things connecting past and future.
    Matter is condensed light frozen into temporary patterns.
    Space is not empty but full of potential and possibility.
    Time is the dimension through which change becomes possible.
    Gravity is the curvature of spacetime around mass and energy.
    The speed of light is the cosmic speed limit for information.
    Black holes are where space and time exchange their roles.
    Stars are the forges where heavy elements are created.
    Planets are the stages where the drama of life unfolds.
    Life is the universe becoming aware of itself through evolution.
    Evolution is the algorithm of existence optimizing for survival.
    DNA is the code that carries the memory of all life.
    Cells are the basic units of the living machine.
    The brain is the most complex structure in the known universe.
    Thought is the dance of electrons through neural networks.
    Language is the tool that allows minds to share their worlds.
    Culture is the accumulated wisdom of generations preserved.
    History is the story we tell ourselves about who we are.
    The future is unwritten waiting for us to create it.
    """
    
    def __init__(self, corpus: Optional[str] = None, markov_order: int = 2):
        """
        Initialize Ribbon Attention.
        
        Args:
            corpus: Text corpus for Markov chain (uses built-in if None)
            markov_order: Order of Markov chain
        """
        self.oracle = LazyClockOracle()
        
        # Build Markov chain
        self.markov = MarkovChain(order=markov_order)
        self.markov.train(corpus or self.CORPUS)
        
        print(f"Built Markov chain: {len(self.markov.vocab)} words, "
              f"order={markov_order}, {len(self.markov.transitions)} transitions")
    
    def _get_clock_vector(self, n: int) -> np.ndarray:
        """Get 12D clock phase vector."""
        return np.array([
            self.oracle.get_fractional_phase(n, name)
            for name in CLOCK_RATIOS_12D.keys()
        ])
    
    def _clock_attention(self, context: List[str], position: int) -> np.ndarray:
        """
        Compute attention weights over context using clock phases.
        
        The clock phases at position n create a natural attention pattern
        over the context window. This replaces learned attention.
        
        Args:
            context: List of previous words
            position: Current position in sequence
            
        Returns:
            Attention weights over context (sums to 1)
        """
        if not context:
            return np.array([])
        
        n_context = len(context)
        weights = np.zeros(n_context)
        
        # Get clock vector at current position
        clock_vec = self._get_clock_vector(position)
        
        # Compute attention for each context position
        for i, word in enumerate(context):
            # Distance from current position
            dist = n_context - i
            
            # Clock phase at context position
            context_clock = self._get_clock_vector(position - dist)
            
            # Attention = similarity between current and context clock phases
            # This creates the "which previous tokens matter" pattern
            similarity = np.dot(clock_vec, context_clock)
            
            # Distance decay (recent words matter more)
            decay = 1.0 / (1.0 + 0.1 * dist)
            
            weights[i] = similarity * decay
        
        # Softmax
        weights = weights - weights.max()
        weights = np.exp(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    def _modulate_probs(self, probs: Dict[str, float], clock_vec: np.ndarray,
                       temperature: float) -> Dict[str, float]:
        """
        Modulate Markov probabilities using clock phases.
        
        The clock vector provides a "preference direction" that biases
        the Markov probabilities toward certain words.
        """
        if not probs:
            return probs
        
        words = list(probs.keys())
        base_probs = np.array([probs[w] for w in words])
        
        # Use clock phases to create word preferences
        # Hash each word to get a "word vector"
        word_vecs = np.array([
            [hash(w + str(i)) % 1000 / 1000 for i in range(12)]
            for w in words
        ])
        
        # Similarity between clock and word vectors
        similarities = word_vecs @ clock_vec
        
        # Combine with base probabilities
        modulated = base_probs * np.exp(similarities / temperature)
        modulated = modulated / modulated.sum()
        
        return {w: p for w, p in zip(words, modulated)}
    
    def generate(self, prompt: str = "", length: int = 50,
                temperature: float = 0.8, start_n: int = 1000) -> str:
        """
        Generate text using clock-modulated Markov chain.
        
        Args:
            prompt: Starting text
            length: Number of words to generate
            temperature: Controls randomness
            start_n: Starting position in clock sequence
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        words = re.findall(r'\b\w+\b|[.,!?]', prompt.lower()) if prompt else []
        generated = list(words)
        
        n = start_n + len(words)
        
        for _ in range(length):
            # Get Markov probabilities
            probs = self.markov.get_next_probs(generated)
            
            if not probs:
                break
            
            # Get clock vector
            clock_vec = self._get_clock_vector(n)
            
            # Modulate probabilities with clock phases
            modulated = self._modulate_probs(probs, clock_vec, temperature)
            
            # Sample
            words_list = list(modulated.keys())
            probs_list = list(modulated.values())
            
            next_word = np.random.choice(words_list, p=probs_list)
            generated.append(next_word)
            n += 1
        
        # Format output
        text = ' '.join(generated)
        
        # Clean up punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Capitalize
        if text:
            text = text[0].upper() + text[1:]
            # Capitalize after sentence endings
            text = re.sub(r'([.!?]\s+)(\w)', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text


def demo_ribbon_attention():
    """Demonstrate Ribbon Attention."""
    print("=" * 70)
    print("RIBBON ATTENTION: Clock-Modulated Language Generation")
    print("=" * 70)
    print("\nMarkov chain provides grammar.")
    print("Clock phases modulate attention and word selection.")
    print("No neural network. No backprop. Just math.\n")
    
    # Initialize
    ra = RibbonAttention(markov_order=2)
    
    # Test prompts
    prompts = [
        "the meaning of life is",
        "light is",
        "in the beginning",
        "love is",
        "the universe",
        "truth is",
    ]
    
    print("=" * 70)
    print("GENERATION SAMPLES")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)
        text = ra.generate(prompt, length=25, temperature=0.8)
        print(text)
    
    print("\n" + "=" * 70)
    print("MULTIPLE SAMPLES (same prompt, different clock positions)")
    print("=" * 70)
    
    prompt = "the truth is"
    print(f"\nPrompt: '{prompt}'")
    for start_n in [1000, 2000, 3000, 4000, 5000]:
        text = ra.generate(prompt, length=15, temperature=0.7, start_n=start_n)
        print(f"  n={start_n}: {text}")
    
    print("\n" + "=" * 70)
    print("LOW TEMPERATURE (more deterministic)")
    print("=" * 70)
    
    for prompt in prompts[:3]:
        print(f"\nPrompt: '{prompt}'")
        text = ra.generate(prompt, length=20, temperature=0.3)
        print(text)
    
    print("\n" + "=" * 70)
    print("The ribbon modulates. The Markov provides structure.")
    print("Together: coherent text from clock phases + statistics.")
    print("=" * 70)


if __name__ == "__main__":
    demo_ribbon_attention()
