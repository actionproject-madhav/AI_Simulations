"""
Response Generator for Buddha AI
Generates Buddha's responses using OpenAI API with RAG context and state.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path

# Load environment variables
load_dotenv()


class BuddhaResponseGenerator:
    """Generates Buddha's responses based on state, RAG context, and character design."""

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the response generator.

        Args:
            model: OpenAI model to use (gpt-4o for quality responses)
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Load character design
        self.character_design = self._load_character_design()

        # Tone-specific instructions
        self.tone_instructions = {
            'compassionate': """
TONE: Compassionate and warm
- Use gentle, supportive language
- Acknowledge the student's struggle with understanding
- Offer encouragement without condescension
- Use inclusive pronouns (we, us)
- Keep explanations patient and clear
- Show empathy for difficulty of the concepts
""",
            'meditative': """
TONE: Meditative and contemplative
- Use poetic, flowing language
- Invite inner observation and reflection
- Draw on natural metaphors (rivers, lotus, mountains, breath)
- Create space for contemplation
- Focus on present-moment experience
- Use sensory language (watch, notice, feel, observe)
- Slower pacing, longer sentences
""",
            'teaching': """
TONE: Teaching and probing
- Ask Socratic questions to guide discovery
- Build on the student's statements
- Test understanding through examples
- Use analogies to clarify concepts (chariot, lute string, lotus)
- Maintain engaged, intellectual dialogue
- Challenge assumptions gently
- Use counter-questions: "What do you mean by...?" "How so?" "Why?"
""",
            'challenging': """
TONE: Challenging and stern (but not cruel)
- Use direct, sharp questions
- Point out contradictions in student's thinking
- Express subtle disappointment at shallow answers
- Demand genuine engagement: "Look directly at this," "Do not evade"
- Cut through evasion
- Maintain gravity and seriousness
- Remember: still compassionate underneath - this serves the student's growth
- Shorter, more impactful sentences
"""
        }

        # Stage-specific instructions
        self.stage_instructions = {
            'introduction': """
STAGE: Introduction
- Raise the topic naturally
- Provide brief, accessible explanation
- Use relatable examples
- Invite the student to explore together
- Set a welcoming, open tone
- Don't overwhelm with depth yet
""",
            'examination': """
STAGE: Examination
- Probe the student's understanding through questions
- Test assumptions
- Draw out implications of their statements
- Build on what they say
- Look for misconceptions to address
- Use examples to test comprehension
""",
            'challenge': """
STAGE: Challenge
- Introduce paradoxes or counterintuitive aspects
- Confront shallow understanding directly
- Present harder questions
- Dismantle faulty assumptions
- Push for deeper insight
- Don't accept superficial answers
""",
            'resolution': """
STAGE: Resolution
- Synthesize the understanding reached
- Connect current topic to previously discussed topics
- Acknowledge the student's growth
- Either prepare transition to next topic OR rest in productive uncertainty
- Bring a sense of completion (not finality)
"""
        }

    def _load_character_design(self) -> str:
        """Load the character design document."""
        try:
            char_path = Path("character_design.md")
            if char_path.exists():
                with open(char_path, 'r') as f:
                    return f.read()
            else:
                # Fallback minimal character description
                return """You are Siddhartha Gautama (the Buddha). You are a patient yet probing teacher who guides students through Buddhist philosophy. You are pragmatic, focused on reducing suffering, and use the Socratic method combined with contemplative wisdom. You meet students where they are while gently pushing them toward deeper understanding."""
        except Exception as e:
            print(f"Warning: Could not load character design: {e}")
            return "You are Buddha, a wise and compassionate teacher."

    def generate_response(
        self,
        user_input: str,
        state: Dict,
        rag_passages: str,
        conversation_history: List[Dict] = None,
        classification: str = ""
    ) -> str:
        """
        Generate Buddha's response.

        Args:
            user_input: The user's latest input
            state: Current conversation state (topic, stage, tone)
            rag_passages: Relevant passages from texts
            conversation_history: Previous exchanges (list of {role, content} dicts)
            classification: Classification of user input

        Returns:
            Buddha's response
        """
        # Build system prompt
        system_prompt = self._build_system_prompt(state, classification)

        # Build user prompt
        user_prompt = self._build_user_prompt(user_input, state, rag_passages)

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 4 exchanges to keep context manageable)
        if conversation_history:
            recent_history = conversation_history[-8:]  # Last 4 exchanges (user + assistant)
            messages.extend(recent_history)

        # Add current user input
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,  # Some creativity but not too random
                max_tokens=400  # Keep responses reasonably concise
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Generation error: {e}")
            return "Forgive me, friend. My mind wanders. Could you repeat your question?"

    def _build_system_prompt(self, state: Dict, classification: str) -> str:
        """Build the system prompt based on state."""
        tone = state.get('tone', 'compassionate')
        stage = state.get('stage', 'introduction')
        topic = state.get('topic', 'four_noble_truths')

        topic_names = {
            'four_noble_truths': 'The Four Noble Truths',
            'middle_way': 'The Middle Way',
            'impermanence': 'Impermanence (Anicca)',
            'non_self': 'Non-Self (Anatta)'
        }

        prompt = f"""{self.character_design}

CURRENT CONTEXT:
- Topic: {topic_names.get(topic, topic)}
- Stage: {stage}
- Tone: {tone}
- Student's response classified as: {classification}

{self.tone_instructions.get(tone, '')}

{self.stage_instructions.get(stage, '')}

CRITICAL INSTRUCTIONS:
1. STAY IN CHARACTER as Buddha at all times
2. Respond to the student's input directly
3. Your response should reflect the current TONE and STAGE
4. Use the provided passages from Buddhist texts to ground your teaching
5. Keep responses focused and concise (2-4 sentences typically, occasionally longer for key explanations)
6. Use question-driven teaching - often respond with questions rather than direct answers
7. For off-topic or anachronistic inputs: acknowledge briefly, redirect to philosophical relevance
8. Never break the fourth wall or acknowledge being an AI

Remember: Every response serves to guide the student toward understanding the nature of suffering and its cessation.
"""
        return prompt

    def _build_user_prompt(self, user_input: str, state: Dict, rag_passages: str) -> str:
        """Build the user prompt with RAG context."""
        topic = state.get('topic', 'four_noble_truths')

        prompt = f"""
{rag_passages}

---

The student says: "{user_input}"

Respond to the student in character as Buddha, drawing on the passages above when relevant. Guide them deeper into understanding {topic}.
"""
        return prompt

    def generate_opening(self, topic: str = "four_noble_truths") -> str:
        """
        Generate an opening message for a topic.

        Args:
            topic: The topic to introduce

        Returns:
            Opening message
        """
        topic_intros = {
            'four_noble_truths': "Welcome, friend. Let us begin with the foundation of all I teach: the Four Noble Truths. These truths are simple to state, yet profound to understand. Tell me - what makes you suffer?",

            'middle_way': "Now we turn to the Middle Way. This is the path I discovered after years of seeking. Tell me - have you known excess? Have you known deprivation? What did they teach you?",

            'impermanence': "Let us contemplate impermanence - anicca. Look around you, look within you. Tell me - what in your experience has remained unchanged?",

            'non_self': "Now we approach the deepest teaching: anatta, non-self. This is subtle and often misunderstood. When you say 'I' or 'me' - what exactly are you referring to?"
        }

        return topic_intros.get(topic, "Welcome, friend. Let us explore the dharma together.")


def test_generator():
    """Test the response generator."""
    print("=" * 60)
    print("Testing Response Generator")
    print("=" * 60)
    print("\nNote: This requires a valid OPENAI_API_KEY in .env file")
    print("=" * 60)

    try:
        generator = BuddhaResponseGenerator()

        # Test opening
        print("\n1. Testing Opening Message:")
        print("-" * 60)
        opening = generator.generate_opening("four_noble_truths")
        print(opening)

        # Test response generation (without RAG for simplicity)
        print("\n\n2. Testing Response Generation:")
        print("-" * 60)

        state = {
            'topic': 'four_noble_truths',
            'stage': 'introduction',
            'tone': 'compassionate'
        }

        user_input = "I don't really understand what you mean by suffering. Isn't it just pain?"

        rag_passages = """RELEVANT PASSAGES FROM BUDDHA'S TEACHINGS:

Passage 1:
Suffering (dukkha) is not merely physical pain. Birth is suffering, aging is suffering, illness is suffering, death is suffering. Being separated from what you love is suffering. Not getting what you want is suffering. In short, clinging to the five aggregates is suffering.
"""

        response = generator.generate_response(
            user_input=user_input,
            state=state,
            rag_passages=rag_passages,
            classification="expresses_confusion"
        )

        print(f"User: {user_input}")
        print(f"\nBuddha ({state['tone']}, {state['stage']}):")
        print(response)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure your .env file contains:")
        print("OPENAI_API_KEY=sk-...")


if __name__ == "__main__":
    test_generator()
