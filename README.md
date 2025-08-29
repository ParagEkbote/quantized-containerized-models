# quantized-containerized-models

**quantized-containerized-models** is aimed at demonstrating how to deploy AI models inside efficient containerized environments. The repository is designed to help you efficiently serve lightweight, optimized models using modern DevOps practices.

To install cog:

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Features

- **Quantization**: Showcases techniques for reducing model size and speeding up inference.
- **Containerization**: Uses [Cog](https://cog.run/), a Docker-based solution for packaging and deploying AI models.
- **Open Source**: Licensed under the Apache License 2.0.

## Active Deployements

- [flux-fast-lora-hotswap](https://replicate.com/paragekbote/flux-fast-lora-hotswap): Based on the following [blog post](https://huggingface.co/blog/lora-fast), the project features the flux.1-dev models with 2 LoRAs which are hot-swapped to reduce generation time and graph breaks. We also quantize the transformer module to nf4 and torch_compile modules for further speed-ups. The code for the same is present [here](https://github.com/ParagEkbote/quantized-containerized-models/tree/58e64c6e652b2f82f10cab42f25c4093a1252974/flux.1-dev).

- [smollm3-3b-smashed](https://replicate.com/paragekbote/smollm3-3b-smashed): This project uses [Pruna](https://github.com/PrunaAI/pruna) to quantize and torch_compile the smollm3-3b model for faster generation and lower VRAM usage. This allows us to having a higher output context window of 16k tokens and hybrid reasoning capabilites. The code for the same is present [here](https://github.com/ParagEkbote/quantized-containerized-models/tree/58e64c6e652b2f82f10cab42f25c4093a1252974/smollm).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---
