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

## 3. **Canary Release Tests**

**Scope:**  
Live, production-facing validation of a newly deployed candidate by exercising the same inference surface used in production (Replicate), without rebuilding or re-running the container locally.

**What they check:**

- The deployed predictor is callable and responds correctly via the Replicate API
- The model loads successfully on real production GPUs
- Optional runtime branches (e.g., LoRA switching, quantization, sparsity, multimodal paths) execute as expected
- Output validity and quality (format, length, degeneration guards, semantic consistency)
- Performance characteristics under real conditions (latency, cold vs warm behavior, relative throughput)

**What they intentionally do *not* check:**

- Local container builds (`cog build`)
- Docker-in-Docker execution
- Full offline benchmarks or stress tests

Those concerns are handled earlier in the pipeline or in dedicated benchmarking workflows.

**Why it matters:**  
Canary release tests catch failures that only appear **after deployment**, such as:
CUDA or driver incompatibilities, provider-side runtime issues, misconfigured optional branches, degraded output quality, or unexpected latency regressions.

By comparing a candidate deployment against a pinned, known-good stable baseline, canary tests provide high-confidence assurance that the model can be safely promoted to production traffic.


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

Canary tests validate the actual runtime environment: the place where most real errors occur.

### **4. Detects optional-path failures**

LoRA loading, sparsity modes, quantization, image paths, and CUDA/CPU fallbacks are all validated.

### **5. CI/CD safety**

The CI/CD pipeline will never ship a broken container, malformed schema or failing deployment.

---
