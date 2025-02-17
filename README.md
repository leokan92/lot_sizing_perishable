# Perishable Items Environment – Test Documentation

This document lays out our no-nonsense tests in a new environment simulating perishable items. If you think managing expired milk is a headache, wait until our PDPPO agent has to deal with it. Here’s the raw deal.

---

## Overview

The perishable items environment is designed to test our PDPPO agents under extra pressure: items expire, waste costs pile up, and decisions matter even more. We’re not here to sugarcoat—if your agent can’t handle spoilage, it’s time to re-evaluate your strategy.

---

## Repository Structure

We've added a new directory, `Perishable`, which parallels our other setups. Its structure is as follows:

```plaintext
├───Perishable
│   ├───agents          # PDPPO implementations tweaked for perishability challenges
│   ├───envs            # Environment definitions with expiration and spoilage dynamics
│   ├───logs            # Detailed logs capturing training and evaluation under perishable constraints
│   ├───results         # Experiment results and figures for quick comparisons
│   └───test_functions  # Scripts for validating the models and generating performance plots/tables
