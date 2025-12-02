## Testing Strategy

![alt text](../docs/assets/hero_img_testing.webp)

This repository uses a **three-tier test plan** designed specifically for Cog/Replicate model deployments.
Each layer validates a different part of the deployment lifecycle, from Python logic to full container execution.

The goal is simple:
**catch failures early, validate real inference and guarantee production-safe deployments.**

---

## 1. **Unit Tests**

**Scope:** Local, fast tests that verify the internal logic of `predict.py` without loading large models.

**What they check:**

* Function signatures and input validation
* Schema defaults and constraints
* Expected errors for invalid inputs
* Lightweight utility functions
* Mocked branches (quantization flags, sparsity flags and hotswap logic)

**Why it matters:**
Unit tests catch deterministic bugs quickly, without requiring GPUs or remote API calls. They form the fastest feedback loop during development.

---

## 2. **Integration Tests**

**Scope:** Tests that call **Replicate’s Prediction API** using the latest deployed version.

**What they check:**

* API contract matches your Cog schema
* Real remote input validation (422 errors, types, defaults)
* Streaming vs. file outputs
* Consistency of actual model results
* That the Replicate deployment responds successfully

**Why it matters:**
The Python model code can behave differently once packaged into a Cog container.
Integration tests validate the *real deployment interface* and ensure the container behaves exactly as the schema declares.

---

## 3. **Deployment Tests**

**Scope:** Full end-to-end validation of the Cog container environment; the same environment used by Replicate for production inference.

**What they check:**

* Container builds correctly (`cog build`)
* Predictor boots and loads the real model
* LoRA / sparsity / quantization branches run in realistic mode
* GPU kernels, memory limits, and runtime stability
* Performance characteristics (latency, throughput, warm/cold behavior)

**Why it matters:**
These tests catch the issues that unit and integration tests *cannot*:
CUDA mismatches, missing dependencies, broken model weights, or optional branches failing only at runtime.

Deployment tests prove that the model will run reliably once shipped.

---

## ⚙ How CI/CD Uses This Plan

### **CI (Pull Requests / Commits)**

Runs:

* **Lint**
* **Unit tests**
* **Security Scanning**

CI verifies code quality, scans for leaked credentials and ensures the API contract and logic of `predict.py` is not broken.

### **CD (Deployments)**

Runs:

* **All tests:** Unit + Integration + Deployment
* Ensures the actual production endpoint behaves correctly

CD provides complete assurance *before pushing new versions to Replicate*.

---

## Benefits

### **1. Fast local development**

Unit tests run in seconds and catch most mistakes before packaging containers.

### **2. Guaranteed API correctness**

Integration tests ensure your schema matches the live Replicate endpoint.

### **3. Reliable production deployments**

Deployment tests validate the actual runtime environment: the place where most real errors occur.

### **4. Detects optional-path failures**

LoRA loading, sparsity modes, quantization, image paths, and CUDA/CPU fallbacks are all validated.

### **5. CI/CD safety**

The CI/CD pipeline will never ship a broken container, malformed schema or failing deployment.

---