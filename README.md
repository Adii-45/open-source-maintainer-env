---
title: Open Source Maintainer Env
emoji: 🏆
colorFrom: green
colorTo: red
sdk: docker
pinned: false
license: apache-2.0
---

# 🛠️ Open Source Maintainer Environment

An interactive, containerized environment simulating the daily workflow of an open-source repository maintainer. Built for agentic evaluation as part of the **Meta x Scaler OpenEnv Hackathon**.

## 🚀 Project Overview
This project provides a robust testing ground for LLM agents. It challenges them to act as repo maintainers by handling tasks such as:
* Triaging and labeling frontend UI bugs.
* Identifying and safely closing duplicate issues.
* Reviewing pull requests and catching algorithmic inefficiencies (e.g., catching O(n^2) time complexity flaws).

## ⚙️ Tech Stack
* **Framework:** FastAPI / Uvicorn
* **Core:** OpenEnv
* **LLM Reasoning:** Meta Llama-3 (via Hugging Face API)
* **Deployment:** Docker & Hugging Face Spaces

## ✅ Validation Status
This environment has been fully containerized and successfully passed the official Meta x Scaler `validate-submission.sh` compliance checks (3/3 Checks Passed). It is fully ready for multi-mode deployment.

## 👥 Team: AI Avengers
* **Rohit Gupta** (Team Lead)
* **Tannaya Supriya**
* **Aditya Kumar Nayak**

---
*Built with ❤️ for the OpenEnv Hackathon Round 1*
