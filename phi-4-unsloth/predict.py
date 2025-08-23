import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from cog import BasePredictor, Input
from unsloth import FastLanguageModel
import soundfile as sf


class Predictor(BasePredictor):
    model: Any  # Change from torch.nn.Module to Any to be more flexible
    tokenizer: Any
    device: torch.device

    def setup(self) -> None:
        model_id: str = "unsloth/orpheus-3b-0.1-ft"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Ensure we have the actual model, not just parameters
        self.model = model
        self.tokenizer = tokenizer

        # Ensure pad token exists (fallback to eos token if missing)
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Debug: Print model type to verify it's correct
        print(f"Model type: {type(self.model)}")
        print(f"Model has generate method: {hasattr(self.model, 'generate')}")

    def predict(self, prompt: str = Input(description="Input prompt")) -> str:
        # Tokenize input
        encoding: Dict[str, torch.Tensor] = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move tensors to device
        inputs: Dict[str, torch.Tensor] = {k: v.to(self.device) for k, v in encoding.items()}

        # Get token IDs safely
        eos_id: Optional[int] = None
        pad_id: Optional[int] = None
        
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            eos_id = int(self.tokenizer.eos_token_id)
        
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            pad_id = int(self.tokenizer.pad_token_id)
        elif eos_id is not None:
            pad_id = eos_id

        # Generate audio tokens with better error handling
        try:
            with torch.no_grad():
                # Verify model has generate method before calling
                if not hasattr(self.model, 'generate'):
                    raise AttributeError(f"Model of type {type(self.model)} does not have a 'generate' method")
                
                generation_kwargs = {
                    **inputs,
                    'max_new_tokens': 125,
                    'do_sample': True,
                    'temperature': 0.7,
                    'top_p': 0.95,
                }
                
                # Only add token IDs if they exist
                if eos_id is not None:
                    generation_kwargs['eos_token_id'] = eos_id
                if pad_id is not None:
                    generation_kwargs['pad_token_id'] = pad_id
                
                audio_tokens = self.model.generate(**generation_kwargs)
                
        except Exception as e:
            return f"Generation failed: {str(e)}. Model type: {type(self.model)}"

        # Process generated tokens
        if isinstance(audio_tokens, (list, tuple)):
            tokens_tensor = audio_tokens[0]
        else:
            tokens_tensor = audio_tokens
            
        # Handle batch dimension
        if tokens_tensor.dim() > 1:
            tokens_tensor = tokens_tensor[0]

        # Decode tokens to waveform
        try:
            if hasattr(self.tokenizer, 'decode_audio'):
                waveform = self.tokenizer.decode_audio(tokens_tensor)
            else:
                return f"Tokenizer does not have decode_audio method. Available methods: {[m for m in dir(self.tokenizer) if not m.startswith('_')]}"
        except Exception as e:
            return f"Audio decoding failed: {str(e)}"

        # Save audio file
        try:
            wav_path: Path = Path(tempfile.gettempdir()) / "output.wav"
            self.save_audio(waveform, wav_path)
            return f"Generated speech saved to {wav_path}"
        except Exception as e:
            return f"Audio saving failed: {str(e)}"

    def save_audio(self, waveform: Union[torch.Tensor, Any], filename: Path) -> None:
        """Save generated waveform to a .wav file."""
        try:
            # Convert torch.Tensor to numpy if necessary
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
            else:
                waveform_np = waveform

            # Get sampling rate with fallback
            sampling_rate: int = getattr(self.tokenizer, "sampling_rate", 22050)
            if not isinstance(sampling_rate, int):
                sampling_rate = 22050
                
            sf.write(str(filename), waveform_np, sampling_rate)
            print(f"Audio saved to {filename}")
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            raise