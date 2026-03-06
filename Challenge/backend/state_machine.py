"""
State Machine for Krishnamurti AI Conversation
Manages topic, stage, and tone transitions based on user interactions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ConversationState:
    """Represents the current state of the conversation."""
    topic: str = "four_noble_truths"  # Current philosophical topic
    stage: str = "introduction"        # introduction/examination/challenge/resolution
    tone: str = "teaching"             # teaching/sad/happy/contemplate
    turn_count: int = 0                # Number of turns in conversation
    topic_turn_count: int = 0          # Number of turns in current topic
    topics_covered: List[str] = None   # Topics already discussed

    def __post_init__(self):
        if self.topics_covered is None:
            self.topics_covered = []

    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationState':
        """Create state from dictionary."""
        return cls(**data)


class KrishnamurtiStateMachine:
    """Manages conversation state transitions for Krishnamurti AI."""

    # Topic configuration
    TOPICS = {
        'four_noble_truths': {
            # Interpreted in Krishnamurti style as the fact of suffering and its roots,
            # not as a doctrinal list.
            'name': 'Suffering and its roots',
            'keywords': ['suffering', 'conflict', 'loneliness', 'comparison', 'fear'],
            'next_topics': ['middle_way', 'impermanence'],
            'difficulty': 1
        },
        'middle_way': {
            # Interpreted as order and balance in daily life, not a Buddhist path.
            'name': 'Order and balance in life',
            'keywords': ['order', 'disorder', 'balance', 'extremes', 'security'],
            'next_topics': ['impermanence', 'non_self'],
            'difficulty': 2
        },
        'impermanence': {
            'name': 'Change and instability',
            'keywords': ['change', 'instability', 'loss', 'time', 'becoming'],
            'next_topics': ['non_self'],
            'difficulty': 3
        },
        'non_self': {
            'name': 'Self and the observer',
            'keywords': ['self', 'observer', 'center', 'ego', 'image'],
            'next_topics': [],  # Terminal node
            'difficulty': 4
        }
    }

    # Stage progression rules
    STAGES = ['introduction', 'examination', 'challenge', 'resolution']

    # Tone transition rules (mapped from classifier categories)
    TONE_TRANSITIONS = {
        # Student is struggling or vulnerable → more "sad"/soft teaching
        'expresses_confusion': 'sad',
        # Active learning and insight → "happy"
        'asks_clarifying_question': 'happy',
        'demonstrates_understanding': 'happy',
        # Deep insight/reflection → "contemplate"
        'insightful_response': 'contemplate',
        # Low-effort or evasive → firmer "teaching" tone
        'minimal_answer': 'teaching',
        'evasive_answer': 'teaching',
        'off_topic': 'teaching',
    }

    def __init__(self):
        """Initialize the state machine."""
        self.state = ConversationState()

    def get_state(self) -> ConversationState:
        """Get current state."""
        return self.state

    def update_tone(self, classification: str) -> str:
        """
        Update tone based on user input classification.

        Args:
            classification: User input classification

        Returns:
            New tone
        """
        # Get new tone from transition rules
        new_tone = self.TONE_TRANSITIONS.get(classification, self.state.tone)

        # Special rules based on current stage
        if self.state.stage == 'introduction':
            # Prefer gentler tones at the beginning
            if new_tone == 'teaching':
                new_tone = 'happy'

        elif self.state.stage == 'challenge':
            # Allow more direct teaching during challenge stage
            if new_tone == 'sad' and classification != 'expresses_confusion':
                new_tone = 'teaching'

        elif self.state.stage == 'resolution':
            # Prefer contemplative or happy in resolution
            if new_tone == 'teaching':
                new_tone = 'contemplate'

        self.state.tone = new_tone
        return new_tone

    def update_stage(self, classification: str, turn_count_in_topic: int) -> str:
        """
        Update conversation stage based on user engagement.

        Args:
            classification: User input classification
            turn_count_in_topic: Number of turns in current topic

        Returns:
            New stage
        """
        current_stage = self.state.stage
        current_idx = self.STAGES.index(current_stage)

        # Progression logic
        if classification == 'demonstrates_understanding':
            # Move forward if demonstrating understanding
            if current_stage == 'introduction' and turn_count_in_topic >= 2:
                new_idx = min(current_idx + 1, len(self.STAGES) - 1)
                self.state.stage = self.STAGES[new_idx]

            elif current_stage == 'examination' and turn_count_in_topic >= 4:
                new_idx = min(current_idx + 1, len(self.STAGES) - 1)
                self.state.stage = self.STAGES[new_idx]

        elif classification == 'insightful_response':
            # Jump forward on insightful responses
            if current_stage == 'introduction':
                self.state.stage = 'examination'
            elif current_stage == 'examination':
                self.state.stage = 'challenge'

        elif classification == 'expresses_confusion':
            # Don't regress, but also don't progress
            pass

        elif classification in ['minimal_answer', 'evasive_answer', 'off_topic']:
            # Stay in current stage or move to challenge
            if current_stage != 'challenge' and turn_count_in_topic >= 3:
                self.state.stage = 'challenge'

        # Natural progression based on turn count
        if turn_count_in_topic >= 6 and current_stage == 'introduction':
            self.state.stage = 'examination'
        elif turn_count_in_topic >= 10 and current_stage == 'examination':
            self.state.stage = 'challenge'
        elif turn_count_in_topic >= 14 and current_stage == 'challenge':
            self.state.stage = 'resolution'

        return self.state.stage

    def should_transition_topic(self, classification: str) -> bool:
        """
        Determine if it's time to transition to a new topic.

        Args:
            classification: User input classification

        Returns:
            True if should transition to new topic
        """
        # Must be in resolution stage
        if self.state.stage != 'resolution':
            return False

        # Must show understanding or insight
        if classification not in ['demonstrates_understanding', 'insightful_response']:
            return False

        # Must have spent enough time on topic
        if self.state.topic_turn_count < 8:
            return False

        # Must have next topics available
        next_topics = self.TOPICS[self.state.topic]['next_topics']
        if not next_topics:
            # Terminal topic reached
            return False

        return True

    def transition_topic(self, classification: str) -> Optional[str]:
        """
        Transition to the next topic if appropriate.

        Args:
            classification: User input classification

        Returns:
            New topic or None if no transition
        """
        if not self.should_transition_topic(classification):
            return None

        # Get next topics
        next_topics = self.TOPICS[self.state.topic]['next_topics']

        # Filter out already covered topics
        available_topics = [t for t in next_topics if t not in self.state.topics_covered]

        if not available_topics:
            # All topics covered, stay on current
            return None

        # Choose next topic (could be made more intelligent)
        # For now, just pick the first available
        new_topic = available_topics[0]

        # Update state
        self.state.topics_covered.append(self.state.topic)
        self.state.topic = new_topic
        self.state.stage = 'introduction'
        self.state.topic_turn_count = 0

        return new_topic

    def update(self, classification: str) -> Dict:
        """
        Update state based on user input classification.

        Args:
            classification: User input classification

        Returns:
            Dictionary with state changes
        """
        old_state = self.state.to_dict()

        # Increment counters
        self.state.turn_count += 1
        self.state.topic_turn_count += 1

        # Update tone
        new_tone = self.update_tone(classification)

        # Update stage
        new_stage = self.update_stage(classification, self.state.topic_turn_count)

        # Check for topic transition
        new_topic = self.transition_topic(classification)

        # Return changes
        changes = {
            'old_topic': old_state['topic'],
            'new_topic': self.state.topic,
            'topic_changed': new_topic is not None,
            'old_stage': old_state['stage'],
            'new_stage': self.state.stage,
            'stage_changed': old_state['stage'] != self.state.stage,
            'old_tone': old_state['tone'],
            'new_tone': self.state.tone,
            'tone_changed': old_state['tone'] != self.state.tone,
            'turn_count': self.state.turn_count,
            'topic_turn_count': self.state.topic_turn_count
        }

        return changes

    def get_topic_info(self, topic: str = None) -> Dict:
        """
        Get information about a topic.

        Args:
            topic: Topic key (defaults to current topic)

        Returns:
            Topic information dictionary
        """
        if topic is None:
            topic = self.state.topic

        return self.TOPICS.get(topic, {})

    def reset(self):
        """Reset state to initial configuration."""
        self.state = ConversationState()


def test_state_machine():
    """Test the state machine."""
    print("=" * 60)
    print("Testing Krishnamurti State Machine")
    print("=" * 60)

    sm = KrishnamurtiStateMachine()

    # Simulate a conversation
    test_sequence = [
        ('expresses_confusion', "User is confused"),
        ('asks_clarifying_question', "User asks question"),
        ('demonstrates_understanding', "User shows understanding"),
        ('demonstrates_understanding', "User shows more understanding"),
        ('insightful_response', "User has insight"),
        ('demonstrates_understanding', "User demonstrates deep understanding"),
        ('demonstrates_understanding', "Keep demonstrating"),
        ('demonstrates_understanding', "Keep demonstrating"),
        ('demonstrates_understanding', "Keep demonstrating"),
        ('demonstrates_understanding', "Ready for next topic"),
    ]

    print("\nInitial State:")
    print(f"  Topic: {sm.state.topic}")
    print(f"  Stage: {sm.state.stage}")
    print(f"  Tone: {sm.state.tone}")

    for i, (classification, description) in enumerate(test_sequence, 1):
        print(f"\n--- Turn {i}: {description} ({classification}) ---")

        changes = sm.update(classification)

        if changes['tone_changed']:
            print(f"  Tone: {changes['old_tone']} -> {changes['new_tone']}")

        if changes['stage_changed']:
            print(f"  Stage: {changes['old_stage']} -> {changes['new_stage']}")

        if changes['topic_changed']:
            print(f"  Topic: {changes['old_topic']} -> {changes['new_topic']}")

        print(f"  Current: {sm.state.topic} / {sm.state.stage} / {sm.state.tone}")

    print("\n" + "=" * 60)
    print("Final State:")
    print(f"  Topic: {sm.state.topic}")
    print(f"  Stage: {sm.state.stage}")
    print(f"  Tone: {sm.state.tone}")
    print(f"  Topics covered: {sm.state.topics_covered}")
    print("=" * 60)


if __name__ == "__main__":
    test_state_machine()
