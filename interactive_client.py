"""
Interactive CLI Client for Interview Agent FastAPI
Provides a complete user interface for managing interview sessions
"""

import sys
import os
import json
import time
import asyncio
from typing import Optional
from fastapi_client import (
    InterviewAgentClient, 
    InterviewSession, 
    SessionConfig, 
    FastAPIClientError,
    simulate_audio_data,
    record_audio
)

class InteractiveClient:
    """Interactive CLI client for the Interview Agent"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client = InterviewAgentClient(server_url)
        self.current_session: Optional[InterviewSession] = None
        self.running = True
    
    def print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print("           INTERVIEW AGENT FASTAPI CLIENT")
        print("="*60)
    
    def print_menu(self):
        """Print main menu options"""
        print("\n📋 MAIN MENU:")
        print("1.  📊 View Available Models")
        print("2.  🆕 Create New Interview Session")
        print("3.  🎤 Start Interview")
        print("4.  🎵 Process Audio (Simulated)")
        print("5.  🎙️  Record and Process Audio")
        print("6.  📝 View Session Information")
        print("7.  💬 View Conversation History")
        print("8.  📋 List All Sessions")
        print("9.  🗑️  Delete Session")
        print("10. ⏹️  End Current Session")
        print("11. 🔄 Test Server Connection")
        print("0.  🚪 Exit")
        print("-" * 60)
    
    def get_user_choice(self, prompt: str = "Enter your choice: ") -> str:
        """Get user input with error handling"""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
    
    def wait_for_enter(self, message: str = "Press Enter to continue..."):
        """Wait for user to press Enter"""
        try:
            input(message)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
    
    def view_available_models(self):
        """View available ASR and LLM models"""
        print("\n🔍 FETCHING AVAILABLE MODELS...")
        try:
            models = self.client.get_available_models()
            
            print("\n📊 ASR MODELS:")
            print("-" * 40)
            for key, model_info in models.get('asr_models', {}).items():
                print(f"{key}. {model_info['name']}")
                print(f"   Description: {model_info['description']}")
                print(f"   Language: {model_info['language']}")
                print()
            
            print("🤖 LLM MODELS:")
            print("-" * 40)
            for key, model_info in models.get('llm_models', {}).items():
                print(f"{key}. {model_info['provider']}/{model_info['model']}")
            print()
            
        except FastAPIClientError as e:
            print(f"❌ Error: {e}")
    
    def create_session(self):
        """Create a new interview session"""
        print("\n🆕 CREATING NEW INTERVIEW SESSION")
        print("-" * 40)
        
        # Get interview topic
        topic = self.get_user_choice("Enter interview topic (e.g., Machine Learning): ")
        if not topic:
            print("❌ Topic cannot be empty")
            return
        
        # Get resume information
        print("\n📄 RESUME CONFIGURATION:")
        print("You can provide:")
        print("- A user ID (UUID) to fetch resume from user profile")
        print("- A direct path to resume in Supabase storage (e.g., 'user_id/filename.pdf')")
        print("- A local file path")
        
        resume_file = self.get_user_choice("Enter user ID, resume path, or local file path: ")
        if not resume_file:
            print("⚠️  No resume information provided. Using default user ID 'user123'")
            resume_file = "user123"
        
        # Get model preferences
        print("\n🔧 MODEL CONFIGURATION:")
        asr_model = self.get_user_choice("ASR model (press Enter for default 'openai/whisper-small'): ")
        if not asr_model:
            asr_model = "openai/whisper-small"
        
        llm_provider = self.get_user_choice("LLM provider (press Enter for default 'ollama'): ")
        if not llm_provider:
            llm_provider = "ollama"
        
        llm_model = self.get_user_choice("LLM model (press Enter for default 'llama3.2:1b'): ")
        if not llm_model:
            llm_model = "llama3.2:1b"
        
        # Create session configuration
        config = SessionConfig(
            interview_topic=topic,
            resume_file=resume_file,
            asr_model=asr_model,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        try:
            print("\n🔄 Creating session...")
            self.current_session = InterviewSession(self.client, config)
            print("✅ Session created successfully!")
            
        except FastAPIClientError as e:
            print(f"❌ Error creating session: {e}")
    
    def start_interview(self):
        """Start the interview"""
        if not self.current_session:
            print("❌ No active session. Create a session first.")
            return
        
        print("\n🎤 STARTING INTERVIEW")
        print("-" * 40)
        
        try:
            print("🔄 Starting interview...")
            response = self.current_session.start()
            
            print(f"\n🤖 AI: {response.response}")
            
            if response.audio_response:
                print("🔊 Audio response available")
                play_audio = self.get_user_choice("Play audio response? (y/n): ").lower()
                if play_audio == 'y':
                    self.client.play_audio_response(response.audio_response)
            
        except FastAPIClientError as e:
            print(f"❌ Error starting interview: {e}")
    
    def process_simulated_audio(self):
        """Process simulated audio"""
        if not self.current_session:
            print("❌ No active session. Create a session first.")
            return
        
        print("\n🎵 PROCESSING SIMULATED AUDIO")
        print("-" * 40)
        
        try:
            # Get simulation parameters
            duration = float(self.get_user_choice("Audio duration (seconds, default 2.0): ") or "2.0")
            frequency = float(self.get_user_choice("Frequency (Hz, default 440): ") or "440")
            
            print(f"🔄 Generating {duration}s audio at {frequency}Hz...")
            audio_data = simulate_audio_data(duration, frequency)
            
            print("🔄 Processing audio...")
            response = self.current_session.respond_to_audio(audio_data, play_audio=False)
            
            if response.transcription:
                print(f"📝 Transcription: {response.transcription}")
            else:
                print("📝 No transcription (likely no speech detected)")
            
            if response.response:
                print(f"🤖 AI Response: {response.response}")
                
                if response.audio_response:
                    play_audio = self.get_user_choice("Play audio response? (y/n): ").lower()
                    if play_audio == 'y':
                        self.client.play_audio_response(response.audio_response)
            else:
                print("🤖 No AI response generated")
                
        except FastAPIClientError as e:
            print(f"❌ Error processing audio: {e}")
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def record_and_process_audio(self):
        """Record and process real audio"""
        if not self.current_session:
            print("❌ No active session. Create a session first.")
            return
        
        print("\n🎙️ RECORDING AND PROCESSING AUDIO")
        print("-" * 40)
        
        try:
            duration = float(self.get_user_choice("Recording duration (seconds, default 5.0): ") or "5.0")
            
            print(f"🎙️ Recording {duration} seconds of audio...")
            print("🎤 Speak now!")
            
            audio_data = record_audio(duration)
            
            print("🔄 Processing audio...")
            response = self.current_session.respond_to_audio(audio_data, play_audio=False)
            
            if response.transcription:
                print(f"📝 Transcription: {response.transcription}")
            else:
                print("📝 No transcription (likely no speech detected)")
            
            if response.response:
                print(f"🤖 AI Response: {response.response}")
                
                if response.audio_response:
                    play_audio = self.get_user_choice("Play audio response? (y/n): ").lower()
                    if play_audio == 'y':
                        self.client.play_audio_response(response.audio_response)
            else:
                print("🤖 No AI response generated")
                
        except FastAPIClientError as e:
            print(f"❌ Error processing audio: {e}")
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def view_session_info(self):
        """View current session information"""
        if not self.current_session:
            print("❌ No active session. Create a session first.")
            return
        
        print("\n📝 SESSION INFORMATION")
        print("-" * 40)
        
        try:
            session_info = self.client.get_session_info()
            
            print(f"🆔 Session ID: {session_info.get('session_id', 'N/A')}")
            print(f"📋 Topic: {session_info.get('interview_topic', 'N/A')}")
            print(f"📊 Status: {session_info.get('status', 'N/A')}")
            print(f"🎤 ASR Model: {session_info.get('asr_model', 'N/A')}")
            print(f"🤖 LLM Model: {json.dumps(session_info.get('llm_model', {}), indent=2)}")
            
            conversation_history = session_info.get('conversation_history', [])
            print(f"💬 Conversation turns: {len(conversation_history)}")
            
        except FastAPIClientError as e:
            print(f"❌ Error getting session info: {e}")
    
    def view_conversation_history(self):
        """View conversation history"""
        if not self.current_session:
            print("❌ No active session. Create a session first.")
            return
        
        print("\n💬 CONVERSATION HISTORY")
        print("-" * 40)
        
        history = self.current_session.get_conversation_history()
        
        if not history:
            print("📭 No conversation history yet")
            return
        
        for i, message in enumerate(history, 1):
            role_icon = "🤖" if message['role'] == 'assistant' else "👤"
            print(f"{i}. {role_icon} {message['role'].title()}: {message['content']}")
            print()
    
    def list_sessions(self):
        """List all sessions"""
        print("\n📋 LISTING ALL SESSIONS")
        print("-" * 40)
        
        try:
            sessions_data = self.client.list_sessions()
            sessions = sessions_data.get('sessions', [])
            
            if not sessions:
                print("📭 No sessions found")
                return
            
            for i, session in enumerate(sessions, 1):
                print(f"{i}. Session ID: {session.get('session_id', 'N/A')}")
                print(f"   Topic: {session.get('interview_topic', 'N/A')}")
                print(f"   Status: {session.get('status', 'N/A')}")
                print(f"   Created: {session.get('created_at', 'N/A')}")
                if session.get('ended_at'):
                    print(f"   Ended: {session.get('ended_at', 'N/A')}")
                print()
                
        except FastAPIClientError as e:
            print(f"❌ Error listing sessions: {e}")
    
    def delete_session(self):
        """Delete a session"""
        print("\n🗑️ DELETE SESSION")
        print("-" * 40)
        
        try:
            sessions_data = self.client.list_sessions()
            sessions = sessions_data.get('sessions', [])
            
            if not sessions:
                print("📭 No sessions to delete")
                return
            
            print("Available sessions:")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. {session.get('session_id', 'N/A')} - {session.get('interview_topic', 'N/A')}")
            
            choice = self.get_user_choice("\nEnter session number to delete (or 'c' to cancel): ")
            if choice.lower() == 'c':
                return
            
            try:
                session_index = int(choice) - 1
                if 0 <= session_index < len(sessions):
                    session_id = sessions[session_index]['session_id']
                    confirm = self.get_user_choice(f"Delete session {session_id}? (y/n): ").lower()
                    if confirm == 'y':
                        result = self.client.delete_session(session_id)
                        print(f"✅ {result.get('message', 'Session deleted')}")
                    else:
                        print("❌ Deletion cancelled")
                else:
                    print("❌ Invalid session number")
            except ValueError:
                print("❌ Invalid input")
                
        except FastAPIClientError as e:
            print(f"❌ Error deleting session: {e}")
    
    def end_session(self):
        """End current session"""
        if not self.current_session:
            print("❌ No active session to end")
            return
        
        print("\n⏹️ ENDING SESSION")
        print("-" * 40)
        
        try:
            confirm = self.get_user_choice("End current session? (y/n): ").lower()
            if confirm == 'y':
                result = self.current_session.end()
                print(f"✅ {result.get('message', 'Session ended')}")
                self.current_session = None
            else:
                print("❌ Session ending cancelled")
                
        except FastAPIClientError as e:
            print(f"❌ Error ending session: {e}")
    
    def test_connection(self):
        """Test server connection"""
        print("\n🔄 TESTING SERVER CONNECTION")
        print("-" * 40)
        
        try:
            self.client._test_connection()
            print("✅ Server connection successful!")
            
            # Test getting models
            models = self.client.get_available_models()
            print(f"✅ Available models: {len(models.get('asr_models', {}))} ASR, {len(models.get('llm_models', {}))} LLM")
            
        except FastAPIClientError as e:
            print(f"❌ Connection failed: {e}")
    
    def run(self):
        """Main application loop"""
        self.print_header()
        
        while self.running:
            self.print_menu()
            
            choice = self.get_user_choice("Enter your choice (0-11): ")
            
            if choice == "0":
                print("\n👋 Goodbye!")
                self.running = False
            elif choice == "1":
                self.view_available_models()
            elif choice == "2":
                self.create_session()
            elif choice == "3":
                self.start_interview()
            elif choice == "4":
                self.process_simulated_audio()
            elif choice == "5":
                self.record_and_process_audio()
            elif choice == "6":
                self.view_session_info()
            elif choice == "7":
                self.view_conversation_history()
            elif choice == "8":
                self.list_sessions()
            elif choice == "9":
                self.delete_session()
            elif choice == "10":
                self.end_session()
            elif choice == "11":
                self.test_connection()
            else:
                print("❌ Invalid choice. Please enter a number between 0-11.")
            
            if self.running:
                self.wait_for_enter()

def main():
    """Main entry point"""
    print("🚀 Starting Interactive Interview Agent Client...")
    
    # Get server URL
    server_url = input("Enter server URL (default: http://localhost:8000): ").strip()
    if not server_url:
        server_url = "http://localhost:8000"
    
    # Create and run interactive client
    client = InteractiveClient(server_url)
    client.run()

if __name__ == "__main__":
    main() 