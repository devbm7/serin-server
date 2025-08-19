# Model Selection Features

## Overview

The conversational AI pipeline now includes interactive model selection features that allow users to choose their preferred ASR (Automatic Speech Recognition) and LLM (Large Language Model) models for each session.

## Available Models

### ASR Models (Whisper)

The system offers 8 different Whisper model options:

| Option | Model Name                        | Description            | Parameters | Use Case                              |
| ------ | --------------------------------- | ---------------------- | ---------- | ------------------------------------- |
| 0      | `openai/whisper-small`          | Default model          | 244M       | Balanced performance                  |
| 1      | `openai/whisper-tiny`           | Fastest, smallest      | 39M        | Quick processing, limited resources   |
| 2      | `openai/whisper-base`           | Small model            | 74M        | Good balance of speed and accuracy    |
| 3      | `openai/whisper-small`          | Medium model (Default) | 244M       | Recommended for most users            |
| 4      | `openai/whisper-medium`         | Large model            | 769M       | Higher accuracy, more resources       |
| 5      | `openai/whisper-large`          | Very large model       | 1550M      | Maximum accuracy, high resource usage |
| 6      | `openai/whisper-large-v3`       | Latest large model     | 1550M      | Best performance, latest improvements |
| 7      | `openai/whisper-large-v3-turbo` | Latest turbo model     | 1550M      | Fastest large model                   |

### LLM Models (Ollama)

The system automatically detects and displays all available Ollama models on your system. Based on your current setup, you have:

- `deepseek-r1:1.5b` (1.1 GB)
- `gemma3:1b` (815 MB)
- `qwen3:0.6b` (522 MB)
- `llama3.2:1b` (1.3 GB) - Default
- `gemma3:latest` (3.3 GB)

## How It Works

1. **Session Start**: When you run the pipeline, you'll be prompted to select both ASR and LLM models
2. **Model Display**: The system shows all available models with descriptions and sizes
3. **User Selection**: Choose models by entering the corresponding numbers
4. **Model Loading**: The selected models are loaded into memory
5. **Session Use**: The selected models are used throughout the session

## Usage

### Running the Main Pipeline

```bash
python pipeline.py
```

You'll see output like:

```
==== Conversational AI CLI ====
Welcome! You'll be prompted to select ASR and LLM models for this session.
Press Enter to start listening. Press Ctrl+C to exit.

==== ASR Model Selection ====
Please select an ASR model for this session:

1. openai/whisper-tiny
    Fastest, smallest model (39M parameters)
    Language: Multilingual

2. openai/whisper-base
    Small model (74M parameters)
    Language: Multilingual

3. openai/whisper-small
    Medium model (244M parameters) - Default
    Language: Multilingual

4. openai/whisper-medium
    Large model (769M parameters)
    Language: Multilingual

5. openai/whisper-large
    Very large model (1550M parameters)
    Language: Multilingual

6. openai/whisper-large-v3
    Latest large model with improved performance
    Language: Multilingual

7. openai/whisper-large-v3-turbo
    Latest large model with improved performance
    Language: Multilingual

0. Use default model (openai/whisper-small)

Enter your choice (0-7): 

==== LLM Model Selection ====
Please select an LLM model for this session:

1. deepseek-r1:1.5b
    Size: 1100.0 MB

2. gemma3:1b
    Size: 815.0 MB

3. qwen3:0.6b
    Size: 522.0 MB

4. llama3.2:1b
    Size: 1300.0 MB

5. gemma3:latest
    Size: 3300.0 MB

0. Use default model (llama3.2:1b)

Enter your choice (0-5): 
```

### Testing the Selection Interfaces

To test the model selection without running the full pipeline:

```bash
# Test ASR model selection
python test_asr_selection.py

# Test LLM model selection
python test_llm_selection.py
```

## Model Selection Guidelines

### ASR Model Selection:

- **Speed Priority**: Choose `whisper-tiny` (1) or `whisper-base` (2)
- **Balanced Performance**: Choose `whisper-small` (3) - recommended default
- **Accuracy Priority**: Choose `whisper-medium` (4) or `whisper-large` (5)
- **Best Performance**: Choose `whisper-large-v3` (6) or `whisper-large-v3-turbo` (7)

### LLM Model Selection:

- **Fast Response**: Choose smaller models like `qwen3:0.6b` or `gemma3:1b`
- **Balanced Performance**: Choose `llama3.2:1b` - recommended default
- **High Quality**: Choose `deepseek-r1:1.5b` or `gemma3:latest`
- **Resource Considerations**: Larger models require more RAM and may be slower

### Resource Considerations:

- **Low-end devices**: Use `whisper-tiny` + `qwen3:0.6b`
- **Standard computers**: Use `whisper-small` + `llama3.2:1b` (default)
- **High-end systems**: Use `whisper-large-v3` + `gemma3:latest`
- **GPU available**: Larger models will benefit significantly from GPU acceleration

## Features

- **Interactive Selection**: User-friendly interface with numbered options
- **Model Information**: Each option shows model size and description
- **Dynamic LLM Detection**: Automatically detects available Ollama models
- **Error Handling**: Graceful fallback to default models on errors
- **Keyboard Interrupt**: Ctrl+C during selection uses default models
- **Loading Feedback**: Progress indicators during model loading
- **Session Persistence**: Selected models are used throughout the session

## Technical Details

### ASR Models:

- Models are downloaded from Hugging Face Hub on first use
- Models are cached locally for subsequent sessions
- GPU acceleration is automatically detected and used when available
- All models support multilingual speech recognition

### LLM Models:

- Models are loaded from your local Ollama installation
- Requires Ollama server to be running on localhost:11434
- Model sizes are displayed in MB for easy comparison
- Fallback to default if Ollama server is not available

## Troubleshooting

### Model Download Issues

If ASR model download fails:

- Check internet connection
- Verify sufficient disk space
- Try selecting a smaller model

### Ollama Connection Issues

If LLM model selection fails:

- Ensure Ollama server is running (`ollama serve`)
- Check if models are installed (`ollama list`)
- Verify Ollama is accessible on localhost:11434

### Memory Issues

If you encounter memory errors:

- Choose smaller models for both ASR and LLM
- Close other applications to free memory
- Consider using CPU instead of GPU

### Performance Issues

If processing is slow:

- Choose smaller models
- Ensure GPU is available for larger models
- Check system resources

## Future Enhancements

Potential improvements for future versions:

- Model performance comparison metrics
- Automatic model selection based on system capabilities
- Per-utterance model switching
- Custom model support
- Model caching management
- Model performance profiling
