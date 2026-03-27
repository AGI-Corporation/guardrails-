# Guardrails Framework - Wiki Images

This directory contains visual assets used throughout the Guardrails Framework wiki.

## Image Index

| Filename | Used In | Description |
|---|---|---|
| `guardrails-python-blueprint.png` | Home, Architecture | Python logo on blueprint construction background - framework overview |
| `guardrails-llm-diagram.png` | Home, Architecture | LLM at center with Chaos/Constant input, Text Output, and Guardrails grid |
| `guardrails-filter-diagram.png` | Home, Adversarial Testing | Funnel with red bad data blocked and blue quality data passing through |
| `guardrails-validators-flow.png` | Home, Governance | Funnel labeled Core Concepts/Validators, The Filter, Compliance Data |
| `guardrails-neural-network.png` | Home, Architecture | Neural network globe with orange/blue streams representing LLM abstraction |
| `guardrails-python-robots.png` | Home, Plugin Dev Guide | Python logo surrounded by robotic arms on blueprint - automation/guardrail enforcement |

## How to Add Images

1. Save your image file with the filename listed above.
2. Upload it to this directory via `git add docs/images/<filename>` or the GitHub web UI.
3. The wiki pages reference these images via:
   ```
   https://raw.githubusercontent.com/AGI-Corporation/guardrails-/main/docs/images/<filename>
   ```
4. Images will render automatically in the wiki once committed.

## Image Descriptions

### guardrails-python-blueprint.png
A blueprint-style architectural drawing showing a city under construction with tower cranes, and the Python logo glowing in the center. Represents the Guardrails Framework as infrastructure for building safe Python-based AI systems.

### guardrails-llm-diagram.png  
A diagram showing a Large Language Model (globe/sphere) at the center receiving inputs labeled "Chaos & Constant" on the left, producing "Text Output" on the right, with a glowing grid labeled "GUARDRAILS" intercepting the output. Represents the core guardrail architecture.

### guardrails-filter-diagram.png
A holographic funnel filter with binary data (0s and 1s) flowing in from the top. Red triangle shapes (bad/harmful data) are blocked by the funnel filter while blue cube shapes (quality/safe data) pass through. Labeled in Chinese: 不良数据 (bad data), 守护机制 (guardrail mechanism), 优质数据 (quality data).

### guardrails-validators-flow.png
A neon blue holographic funnel with three labeled stages: "Core Concepts / Validators" at the top, "The Filter" in the middle, and "Compliance Data" at the bottom. Red shapes are rejected (left), blue cube shapes pass through (right). Represents the validator pipeline.

### guardrails-neural-network.png
A wireframe sphere (neural network globe) with orange rays entering from the left and blue grid lines extending to the right. Represents the LLM abstraction layer that allows provider-agnostic guardrail policies.

### guardrails-python-robots.png
An isometric blueprint showing the Python logo on a glowing pedestal surrounded by robotic arms and safety barriers/guardrails. Represents the Plugin System's automated enforcement capabilities.
