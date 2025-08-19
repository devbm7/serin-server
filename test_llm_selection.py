#!/usr/bin/env python3
"""
Test script for LLM model selection functionality.
This script demonstrates the LLM model selection interface without running the full pipeline.
"""

import requests
from rich.console import Console
from rich.theme import Theme

# Rich console for pretty CLI output
custom_theme = Theme({
    "user": "bold cyan",
    "assistant": "bold magenta",
    "prompt": "bold yellow",
    "info": "dim white"
})
console = Console(theme=custom_theme)

def get_available_ollama_models():
    """Get list of available Ollama models from the server."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            return {model['name']: model.get('size', 'Unknown') for model in models}
        else:
            console.print("[bold red]Failed to fetch Ollama models from server[/bold red]")
            return {}
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to Ollama server: {e}[/bold red]")
        return {}

def select_llm_model():
    """Display available LLM models and let user select one."""
    console.print("\n[bold green]==== LLM Model Selection ====[/bold green]")
    console.print("[info]Please select an LLM model for this session:[/info]\n")
    
    # Get available models from Ollama
    available_models = get_available_ollama_models()
    
    if not available_models:
        console.print("[bold red]No Ollama models found or server not running.[/bold red]")
        console.print("[info]Using default model: llama3.2:1b[/info]\n")
        return "llama3.2:1b"
    
    # Display available models
    model_choices = {}
    for i, (model_name, model_size) in enumerate(available_models.items(), 1):
        model_choices[str(i)] = model_name
        size_mb = int(model_size) / (1024 * 1024) if model_size != 'Unknown' else 0
        size_str = f"{size_mb:.1f} MB" if size_mb > 0 else "Unknown size"
        console.print(f"[bold]{i}.[/bold] {model_name}")
        console.print(f"    Size: {size_str}\n")
    
    console.print("[bold]0.[/bold] Use default model (llama3.2:1b)\n")
    
    while True:
        try:
            choice = input("[prompt]Enter your choice (0-{}): [/prompt]".format(len(available_models))).strip()
            if choice == "0":
                console.print("[bold green]Using default LLM model: llama3.2:1b[/bold green]\n")
                return "llama3.2:1b"
            elif choice in model_choices:
                selected_model = model_choices[choice]
                console.print(f"[bold green]Selected LLM model: {selected_model}[/bold green]\n")
                return selected_model
            else:
                console.print(f"[bold red]Invalid choice. Please enter a number between 0-{len(available_models)}.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Using default model: llama3.2:1b[/bold red]\n")
            return "llama3.2:1b"
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[bold red]Using default model: llama3.2:1b[/bold red]\n")
            return "llama3.2:1b"

def main():
    """Test the LLM model selection functionality."""
    console.print("\n[bold green]==== LLM Model Selection Test ====[/bold green]")
    console.print("[info]This is a test of the LLM model selection interface.[/info]")
    console.print("[info]The selected model will be displayed but not actually used.[/info]\n")
    
    try:
        selected_model = select_llm_model()
        console.print(f"[bold blue]Test completed! Selected model: {selected_model}[/bold blue]")
        console.print("[info]In the actual pipeline, this model would now be used for text generation.[/info]")
    except KeyboardInterrupt:
        console.print("\n[bold red]Test interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Test failed with error: {e}[/bold red]")

if __name__ == "__main__":
    main() 