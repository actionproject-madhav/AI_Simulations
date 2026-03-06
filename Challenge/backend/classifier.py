"""
LLM-based Classifier for Krishnamurti AI
Classifies user inputs to drive state machine transitions.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import Dict

# Load environment variables
load_dotenv()


class UserInputClassifier:
    """Classifies user inputs using OpenAI API."""

    # Classification categories
    CATEGORIES = {
        'demonstrates_understanding': 'User shows understanding of the philosophical concept being discussed',
        'expresses_confusion': 'User indicates they are confused, lost, or do not understand',
        'insightful_response': 'User offers a surprisingly deep or insightful perspective',
        'asks_clarifying_question': 'User asks a genuine question seeking clarification',
        'minimal_answer': 'User gives a very brief, superficial, or evasive response',
        'evasive_answer': 'User avoids engaging with the question or deflects',
        'off_topic': 'User brings up something completely unrelated or anachronistic',
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the classifier.

        Args:
            model: OpenAI model to use (gpt-4o-mini is cost-effective for classification)
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _build_classification_prompt(self, user_input: str, context: str = "") -> str:
        """
        Build the classification prompt.

        Args:
            user_input: The user's input to classify
            context: Optional context about current topic

        Returns:
            Classification prompt
        """
        categories_desc = "\n".join([f"- {cat}: {desc}" for cat, desc in self.CATEGORIES.items()])

        prompt = f"""You are classifying user responses in a philosophical dialogue with J. Krishnamurti.

CLASSIFICATION CATEGORIES:
{categories_desc}

{f"CURRENT TOPIC: {context}" if context else ""}

USER INPUT: "{user_input}"

Analyze the user's input and select the SINGLE MOST APPROPRIATE category from the list above.
Return ONLY the category name, nothing else."""

        return prompt

    def classify(self, user_input: str, context: str = "") -> str:
        """
        Classify a user input.

        Args:
            user_input: The user's text input
            context: Optional context (e.g., current topic)

        Returns:
            Classification category
        """
        if not user_input or len(user_input.strip()) < 2:
            return 'minimal_answer'

        try:
            prompt = self._build_classification_prompt(user_input, context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Return only the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic for classification
                max_tokens=50
            )

            classification = response.choices[0].message.content.strip().lower()

            # Validate classification
            if classification in self.CATEGORIES:
                return classification
            else:
                # Default to minimal_answer if invalid response
                print(f"Warning: Invalid classification '{classification}', defaulting to 'minimal_answer'")
                return 'minimal_answer'

        except Exception as e:
            print(f"Classification error: {e}")
            return 'minimal_answer'

    def classify_with_reasoning(self, user_input: str, context: str = "") -> Dict:
        """
        Classify with reasoning explanation.

        Args:
            user_input: The user's text input
            context: Optional context

        Returns:
            Dictionary with classification and reasoning
        """
        try:
            categories_desc = "\n".join([f"- {cat}: {desc}" for cat, desc in self.CATEGORIES.items()])

            prompt = f"""You are classifying user responses in a philosophical dialogue with J. Krishnamurti.

CLASSIFICATION CATEGORIES:
{categories_desc}

{f"CURRENT TOPIC: {context}" if context else ""}

USER INPUT: "{user_input}"

Analyze the user's input and respond in JSON format:
{{
    "category": "the_category_name",
    "reasoning": "brief explanation for why this category was chosen"
}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Respond in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate category
            if result.get('category') not in self.CATEGORIES:
                result['category'] = 'minimal_answer'
                result['reasoning'] = 'Invalid category returned, defaulting to minimal_answer'

            return result

        except Exception as e:
            print(f"Classification error: {e}")
            return {
                'category': 'minimal_answer',
                'reasoning': f'Error during classification: {str(e)}'
            }


def test_classifier():
    """Test the classifier with sample inputs."""
    print("=" * 60)
    print("Testing User Input Classifier")
    print("=" * 60)

    classifier = UserInputClassifier()

    # Test inputs with expected classifications
    test_cases = [
        ("I don't understand what you mean by suffering", "four_noble_truths"),
        ("So attachment to impermanent things causes suffering?", "four_noble_truths"),
        ("Wow, I never thought about it that way. The self is just changing phenomena!", "non_self"),
        ("Could you explain that again?", "middle_way"),
        ("yeah", ""),
        ("ok", ""),
        ("What do you think about cryptocurrency?", ""),
        ("Suffering is like... when things are bad", "four_noble_truths"),
    ]

    print("\nClassification Tests:")
    print("=" * 60)

    for user_input, context in test_cases:
        print(f"\nInput: \"{user_input}\"")
        if context:
            print(f"Context: {context}")

        # Get classification with reasoning
        result = classifier.classify_with_reasoning(user_input, context)

        print(f"Category: {result['category']}")
        print(f"Reasoning: {result['reasoning']}")

    return classifier


if __name__ == "__main__":
    test_classifier()
