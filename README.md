tfcontracts
========================

Contract-based programming for tensorflow.

## Overview

This library provides a collection of tools to concisely describe constraints on function types / parameters / return values via "contracts". Contracts are enforced at runtime, with the goal of revealing inconsistencies to the user as early as possible. The focus is specifically on simplifying tensorflow-based development.

Roughly speaking, the intention is to provide the following:
- Concise input checking -- primarily oriented towards checking input dimensions, dtypes, and domains (e.g. halfspaces, bounded intervals, objects in `SE(2)`/`SE(3)` manifolds, etc).
- Focus on verifying function pre-conditions and post-conditions.
- Validation of input values.

## Contact

karasev00@gmail.com
