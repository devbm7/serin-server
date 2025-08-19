import json
import os
import csv
import litellm
from rich.console import Console
from rich.panel import Panel

console = Console()

# --- Configuration ---
SESSION_FILE_PATH = 'sessions/session.json'
RESULTS_CSV_PATH = 'evaluation_results.csv'

# Configure LiteLLM
os.environ['LITELLM_LOG'] = 'ERROR'

class InterviewEvaluator:
    def __init__(self, model_info={"provider": "ollama", "model": "llama3.2:1b"}):
        self.model_info = model_info
        self.api_base = "http://localhost:11434"

    def evaluate_conversation(self, topic, conversation_history):
        """Uses an LLM to evaluate a single interview conversation."""
        console.print(f"\n[info]Evaluating interview on topic: [bold]{topic}[/bold]...[/info]")

        # Filter out the initial system and user prompts to get the core conversation
        core_conversation = [msg for msg in conversation_history if msg['role'] in ('assistant', 'user')]
        conversation_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in core_conversation])

        prompt = f"""As a senior technical recruiter, your task is to evaluate the following interview conversation.

**Interview Topic:** {topic}

**Conversation History:**
---
{conversation_text}
---

**Evaluation Criteria:**
1.  **Relevance (1-10):** How relevant were the candidate's answers to the questions?
2.  **Clarity (1-10):** How clearly did the candidate articulate their thoughts?
3.  **Technical Depth (1-10):** How well did the candidate demonstrate deep technical knowledge on the topic?

**Instructions:**
Provide a score for each criterion and a brief, overall summary of the candidate's performance. Format your response as a JSON object with the keys: "relevance_score", "clarity_score", "technical_depth_score", and "summary".

**Example JSON Output:**
{{
  "relevance_score": 8,
  "clarity_score": 7,
  "technical_depth_score": 9,
  "summary": "The candidate demonstrated strong technical knowledge and provided relevant answers, though some explanations could have been more concise."
}}
"""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = litellm.completion(
                model=f"{self.model_info['provider']}/{self.model_info['model']}",
                messages=messages,
                api_base=self.api_base if self.model_info['provider'] == 'ollama' else None
            )
            content = response["choices"][0]["message"]["content"]
            # The response might be wrapped in markdown, so we extract the JSON part
            json_response_str = content[content.find('{'):content.rfind('}')+1]
            evaluation = json.loads(json_response_str)
            console.print(Panel(f"[bold green]✓ Evaluation successful.[/bold green]", title="Status"))
            return evaluation
        except Exception as e:
            console.print(f"[bold red]Error during LLM evaluation: {e}[/bold red]")
            return None

def main():
    console.print("[bold green]==== Interview Evaluation Script ====[/bold green]")

    # Load session data
    try:
        with open(SESSION_FILE_PATH, 'r', encoding='utf-8') as f:
            all_sessions = json.load(f)
        console.print(f"[info]Loaded {len(all_sessions)} interview sessions from [bold]{SESSION_FILE_PATH}[/bold][/info]")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Session file not found at {SESSION_FILE_PATH}[/bold red]")
        return
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Could not decode JSON from {SESSION_FILE_PATH}[/bold red]")
        return

    evaluator = InterviewEvaluator()
    evaluation_results = []

    for session in all_sessions:
        session_time = session.get("time")
        session_data = session.get("session_data", {})
        topic = session_data.get("interview_topic", "N/A")
        history = session_data.get("conversation_history", [])

        if not history:
            console.print(f"[yellow]Skipping session from {session_time} due to empty conversation history.[/yellow]")
            continue

        evaluation = evaluator.evaluate_conversation(topic, history)

        if evaluation:
            evaluation_results.append({
                "session_time": session_time,
                "topic": topic,
                "relevance_score": evaluation.get("relevance_score", "N/A"),
                "clarity_score": evaluation.get("clarity_score", "N/A"),
                "technical_depth_score": evaluation.get("technical_depth_score", "N/A"),
                "summary": evaluation.get("summary", "N/A")
            })

    # Save results to CSV
    if not evaluation_results:
        console.print("[yellow]No evaluations were generated. Exiting.[/yellow]")
        return

    try:
        with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["session_time", "topic", "relevance_score", "clarity_score", "technical_depth_score", "summary"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_results)
        console.print(f"\n[bold green]✓ Evaluation results saved to [bold]{RESULTS_CSV_PATH}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving results to CSV: {e}[/bold red]")

if __name__ == "__main__":
    main()
