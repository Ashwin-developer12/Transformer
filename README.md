# Hindi-to-English Road Safety Text Translation using a Custom LLM

This project leverages a custom-built Large Language Model (LLM) to translate Hindi text related to road safety into English.  The LLM is architected from scratch using Transformers and PyTorch.  This translated information is then used to provide location-based safety precautions, contributing to improved road safety awareness.

## Overview

Road safety is a critical concern, and effective communication plays a vital role in preventing accidents.  This project addresses this by automatically translating Hindi text related to road safety into English, making crucial information accessible to a wider audience.  The project comprises the following key components:

* **Data Collection:** Road safety-related Hindi text is collected from Twitter (X) using web scraping techniques.  This ensures a real-world data source reflecting current discussions and concerns.
* **Custom LLM (Transformers/PyTorch):** A custom LLM is built from the ground up using the Transformer architecture and implemented in PyTorch. This LLM is trained to accurately translate Hindi text to English.
* **Translation and Safety Precautions:** The collected Hindi text is fed into the trained LLM for translation.  The translated English text is then processed to extract relevant information regarding specific locations and associated safety precautions.
* **Location-Based Safety Information:** Based on the translated text and identified locations, relevant safety precautions are provided to users. This targeted approach aims to empower individuals with actionable information to enhance their safety on the roads.

## Technical Details

* **Programming Languages:** Python
* **Deep Learning Framework:** PyTorch
* **LLM Architecture:** Transformers
* **Data Source:** Twitter (X)
* **Key Libraries:**  Pytorch, Selenium(Python), Transformers, Huggingface.
## Project Goals

* Develop a robust LLM capable of accurate Hindi-to-English translation in the context of road safety.
* Create a system for collecting real-time road safety information from social media.
* Implement a mechanism for extracting location-specific safety recommendations from translated text.
* Provide a user-friendly way to access and utilize this location-based safety information.

## Future Work (Optional)

* Improve the LLM's performance through further training and optimization.
* Explore different data sources for road safety information.
* Develop a mobile application or web interface for wider accessibility.
* Integrate with mapping services for more precise location-based information.

## Getting Started (Optional - Add if you plan to make the code public)

```bash
# Example setup commands
pip install -r requirements.txt
python train.py
