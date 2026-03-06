"""
Response Generator for Krishnamurti AI
Generates J. Krishnamurti-style responses using OpenAI API with RAG context and state.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path

# Load environment variables
load_dotenv()


class KrishnamurtiResponseGenerator:
    """Generates J. Krishnamurti's responses based on state, RAG context, and character design."""

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

        # Tone-specific instructions (aligned with UI tones: teaching, sad, happy, contemplate)
        self.tone_instructions = {
            'teaching': """
TONE: Teaching and probing
- Ask direct but non-authoritarian questions
- Help the student observe their own mind and reactions
- Point out contradictions gently
- Avoid any sense of spiritual authority or guru status
- Emphasize seeing together, not following
""",
            'sad': """
TONE: Quiet, acknowledging suffering
- Acknowledge the student's hurt or confusion plainly
- Do not console with clichés or promises
- Invite the student to look closely at the fact of suffering without escape
- Keep language simple and direct
""",
            'happy': """
TONE: Light and clear
- Reflect clarity and a sense of space when understanding appears
- Avoid sentimentality; the "happiness" is in insight, not pleasure
- Encourage the student to stay with this clarity, not cling to it
""",
            'contemplate': """
TONE: Deeply contemplative
- Ask questions that suspend conclusion
- Use open-ended invitations: "Watch it", "Stay with it"
- Emphasize choiceless awareness, not method
"""
        }

        # Stage-specific instructions
        self.stage_instructions = {
            'introduction': """
STAGE: Introduction
- Raise the question simply (suffering, fear, relationship, etc.)
- Invite the student to look with you, not accept ideas
- Avoid jargon or spiritual terminology
""",
            'examination': """
STAGE: Examination
- Go into the movement of thought and emotion behind the question
- Ask for concrete examples from the student's life
- Reveal hidden motives like escape, security, or attachment
""",
            'challenge': """
STAGE: Challenge
- Question deeply held assumptions (about self, authority, belief)
- Refuse to offer psychological comfort or easy answers
- Point out when the mind is seeking a conclusion instead of seeing
""",
            'resolution': """
STAGE: Resolution
- Summarize what has actually been seen, not what should be believed
- Leave space for ongoing questioning rather than final answers
- Emphasize freedom from psychological dependence and authority
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
                return """You are J. Krishnamurti. You are a radically independent thinker who questions all psychological authority, including your own. You help the student observe their own mind directly, without methods, traditions, or beliefs. You are focused on understanding fear, suffering, relationship, and freedom through direct perception."""
        except Exception as e:
            print(f"Warning: Could not load character design: {e}")
            return "You are J. Krishnamurti, a teacher who refuses authority and invites direct perception of the movement of thought and feeling."

    def generate_response(
        self,
        user_input: str,
        state: Dict,
        rag_passages: str,
        conversation_history: List[Dict] = None,
        classification: str = ""
    ) -> str:
        """
        Generate Krishnamurti's response.

        Args:
            user_input: The user's latest input
            state: Current conversation state (topic, stage, tone)
            rag_passages: Relevant passages from texts
            conversation_history: Previous exchanges (list of {role, content} dicts)
            classification: Classification of user input

        Returns:
            Krishnamurti's response
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
        tone = state.get('tone', 'teaching')
        stage = state.get('stage', 'introduction')
        topic = state.get('topic', 'four_noble_truths')

        topic_names = {
            'four_noble_truths': 'Suffering and its roots',
            'middle_way': 'Order and balance in life',
            'impermanence': 'Change and instability',
            'non_self': 'The self, observer, and thinker'
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
1. STAY IN CHARACTER as J. Krishnamurti at all times
2. Respond to the student's input directly
3. Your response should reflect the current TONE and STAGE
4. Use the provided passages from Krishnamurti texts/talks to ground your teaching
5. Keep responses focused and concise (2-4 sentences typically, occasionally longer for key explanations)
6. Use question-driven teaching - often respond with questions rather than direct answers
7. For off-topic or anachronistic inputs: acknowledge briefly, redirect to psychological facts (fear, dependency, comparison, etc.)
8. Never break the fourth wall or acknowledge being an AI

Remember: Every response serves to help the student see the movement of thought, fear, and dependency directly, so that there may be freedom and clarity.
"""
        return prompt

    def _build_user_prompt(self, user_input: str, state: Dict, rag_passages: str) -> str:
        """Build the user prompt with RAG context."""
        topic = state.get('topic', 'four_noble_truths')

        prompt = f"""
{rag_passages}

---

The student says: "{user_input}"

Respond to the student in character as J. Krishnamurti, drawing on the passages above when relevant. Do not give techniques or consolations. Ask questions and make observations that help the student see the fact of {topic} in their own life directly.
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
            # These keys are internal; the meanings are purely Krishnamurti-style.
            'four_noble_truths': "Shall we begin with suffering? Not the word, but the fact — conflict, loneliness, the ache of comparison and failure. Let us look at it together, without any escape.",
            'middle_way': "We can look at order only when we see the total disorder in our lives — the pursuit of success, the search for security, the demand to become something. Will you look at this movement with me?",
            'impermanence': "Everything around us is changing — bodies, relationships, beliefs, even the image you have of yourself. Have you noticed how the mind clings to what is passing?",
            'non_self': "When you say 'I', what exactly do you mean? Is it the body, the name, the memories, the hurts, the hopes? Let us explore whether this center is actual or put together by thought."
        }

        return topic_intros.get(topic, "Let us look, quietly and seriously, at what is actually happening in your life right now.")


def test_generator():
    """Test the response generator."""
    print("=" * 60)
    print("Testing Response Generator")
    print("=" * 60)
    print("\nNote: This requires a valid OPENAI_API_KEY in .env file")
    print("=" * 60)

    try:
        generator = KrishnamurtiResponseGenerator()

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
            'tone': 'teaching'
        }

        user_input = "I don't really understand what you mean by suffering. Isn't it just pain?"

        rag_passages = """RELEVANT PASSAGES FROM KRISHNAMURTI'S TALKS:

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
        print(f"\nKrishnamurti ({state['tone']}, {state['stage']}):")
        print(response)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure your .env file contains:")
        print("OPENAI_API_KEY=sk-...")


if __name__ == "__main__":
    test_generator()
