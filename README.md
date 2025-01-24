# Decentralized Multi-Biped Controller (DecMBC)

Payload transport over varying terrain via highly configurable multi-biped robot carriers.

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red)](https://www.youtube.com/watch?v=2sJQCBaYKsw)
[![Paper](https://img.shields.io/badge/Read-Paper-blue)](https://arxiv.org/abs/2406.17279)
[![Website](https://img.shields.io/badge/Visit-Website-green)](https://decmbc.github.io/)

## Overview

This project implements a decentralized control strategy for multi-legged robots, building upon the work from [RoadRunner](https://github.com/osudrl/roadrunner). Please follow the RoadRunner repository for environment setup. 

## Features

- Decentralized control policy for multi-biped systems
- Support for both flat and uneven terrain navigation
- Pre-trained models for immediate testing
- Customizable terrain configurations

## Quick Start

### Prerequisites
 Follow instructions from [RoadRunner](https://github.com/osudrl/roadrunner)

### Evaluation

1. **Flat Terrain Evaluation**
   ```bash
   ./scripts/eval_cassiepede.sh
   ```

2. **Uneven Terrain Evaluation**
   ```bash
   ./scripts/eval_cassiepede_terrain.sh
   ```

3. **Custom Terrain Configuration**
   
   Modify the terrain parameters in the evaluation scripts:
   ```bash
   # Example: Change terrain index
   --terrain 2  # Choose from available terrain types (0-5)
   ```