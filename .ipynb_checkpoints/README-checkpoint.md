# UWPlasma AI Assistant (vmec_jax Prototype)

This project is a simple AI-assisted interface for vmec_jax.

It uses an Ollama model (e.g., llama3). Please make sure Ollama is installed and running before using the assistant.

## What it does
- Explains how to run vmec_jax from the README
- Describes the workflow from input file to wout output using the example script
- Interprets key output values from simulation results

## How it works
The assistant routes user questions to specific files:
- vmec_readme.md → running commands and installation
- showcase_axisym_input_to_wout.py → workflow
- input.circular_tokamak → input values
- wout_summary.txt → output values

## Usage

Run the assistant:

```bash
python3 assistant.py