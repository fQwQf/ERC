"""
Prompt Templates for Multi-Task Fine-Grained Emotion Recognition
"""

from typing import List, Optional
from dataclasses import dataclass

from ..data.emotion_taxonomy import TAXONOMY


@dataclass
class PromptTemplate:
    """Multi-task prompt template for emotion recognition"""
    
    system_instruction: str = "You are an expert emotion recognition assistant. Analyze the dialogue and identify the emotion of the target utterance from the standard 28-class taxonomy."

    emotion_list: str = "Standard emotions: joy, excitement, happiness, gratitude, pride, relief, hope, love, caring, desire, optimism, amusement, sadness, anger, fear, anxiety, disgust, shame, guilt, disappointment, frustration, grief, loneliness, jealousy, neutral, surprise, confusion, curiosity, nervousness."

    @staticmethod
    def format_dialogue_history(history: List[str], max_turns: int = 5) -> str:
        """Format dialogue history for prompt"""
        if not history:
            return "[No previous context]"
        
        recent_history = history[-max_turns:]
        formatted = []
        for i, utterance in enumerate(recent_history):
            formatted.append(f"Turn {i+1}: {utterance}")
        return "\n".join(formatted)
    
    @staticmethod
    def format_demonstration(
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        speaker: str = "Speaker",
        prev_impact: Optional[str] = None,
    ) -> str:
        """Format a single demonstration example"""
        history_str = PromptTemplate.format_dialogue_history(dialogue_history)
        
        demo = f"""Example:
Dialogue History:
{history_str}

Target Utterance: {target_utterance}

Analysis:
- Emotion: {emotion}
- Speaker: {speaker}"""
        
        if prev_impact:
            demo += f"\n- Impact of previous utterance: {prev_impact}"
        
        return demo
    
    @staticmethod
    def format_query(
        dialogue_history: List[str],
        target_utterance: str,
        prev_impact: Optional[str] = None,
    ) -> str:
        """Format the query (input to be predicted)"""
        history_str = PromptTemplate.format_dialogue_history(dialogue_history)
        
        # If prev_impact is provided, add it to the dialogue history as context
        # This helps the model understand the emotional context of the conversation
        if prev_impact:
            history_str += f"\n\n[Context: {prev_impact}]"
        
        return f"""Now analyze this dialogue:

Dialogue History:
{history_str}

Target Utterance: {target_utterance}

Provide your analysis in the following format:
- Emotion: [your emotion prediction]
- Speaker: [speaker identification]
- Impact: [how previous context influenced the emotion]"""
    
    def build_full_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        demonstrations: Optional[List[dict]] = None,
        include_system: bool = True,
        prev_impact: Optional[str] = None,
    ) -> str:
        """Build complete prompt with optional demonstrations"""
        parts = []
        
        # System instruction
        if include_system:
            parts.append(f"<|im_start|>system\n{self.system_instruction}\n\n{self.emotion_list}<|im_end|>")
        
        # Demonstrations (retrieved examples)
        if demonstrations:
            demo_parts = []
            for demo in demonstrations[:3]:  # Max 3 demonstrations
                demo_str = self.format_demonstration(
                    dialogue_history=demo.get("dialogue_history", []),
                    target_utterance=demo.get("target_utterance", ""),
                    emotion=demo.get("emotion", "neutral"),
                    speaker=demo.get("speaker", "Speaker"),
                    prev_impact=demo.get("impact"),
                )
                demo_parts.append(demo_str)
            
            if demo_parts:
                parts.append(f"<|im_start|>user\nHere are some examples:\n\n" + "\n\n".join(demo_parts) + "<|im_end|>")
        
        # Query
        query = self.format_query(dialogue_history, target_utterance, prev_impact)
        parts.append(f"<|im_start|>user\n{query}<|im_end|>")
        
        # Assistant response start
        parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)
    
    def build_training_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        speaker: str,
        prev_impact: Optional[str] = None,
        demonstrations: Optional[List[dict]] = None,
    ) -> str:
        """Build prompt for training with target response"""
        prompt = self.build_full_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            demonstrations=demonstrations,
            prev_impact=prev_impact,
        )
        
        # Build target response - ALWAYS include Impact field for consistency
        response = f"- Emotion: {emotion}\n- Speaker: {speaker}"
        if prev_impact:
            response += f"\n- Impact: {prev_impact}"
        else:
            # For first utterance or standalone, provide a default standalone impact
            response += "\n- Impact: This is a standalone statement without prior emotional context."
        response += "<|im_end|>"
        
        return prompt + response


class EmotionPromptBuilder:
    """Builder class for creating prompts with retrieval augmentation"""
    
    def __init__(self, use_retrieval: bool = True, top_k: int = 3):
        self.template = PromptTemplate()
        self.use_retrieval = use_retrieval
        self.top_k = top_k
    
    def build_inference_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        retrieved_examples: Optional[List[dict]] = None,
        prev_impact: Optional[str] = None,
    ) -> str:
        """Build prompt for inference"""
        demonstrations = None
        if self.use_retrieval and retrieved_examples:
            demonstrations = retrieved_examples[:self.top_k]
        
        return self.template.build_full_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            demonstrations=demonstrations,
            prev_impact=prev_impact,
        )
    
    def build_training_prompt(
        self,
        dialogue_history: List[str],
        target_utterance: str,
        emotion: str,
        speaker: str,
        prev_impact: Optional[str] = None,
        retrieved_examples: Optional[List[dict]] = None,
    ) -> str:
        """Build prompt for training"""
        demonstrations = None
        if self.use_retrieval and retrieved_examples:
            demonstrations = retrieved_examples[:self.top_k]
        
        return self.template.build_training_prompt(
            dialogue_history=dialogue_history,
            target_utterance=target_utterance,
            emotion=emotion,
            speaker=speaker,
            prev_impact=prev_impact,
            demonstrations=demonstrations,
        )


def parse_model_output(output: str) -> dict:
    """Parse model output to extract predictions"""
    result = {
        "emotion": "neutral",
        "speaker": "Unknown",
        "impact": None,
    }
    
    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Handle both "- Emotion:" and "Emotion:"
        if line.startswith("- Emotion:") or line.startswith("Emotion:"):
            emotion_raw = line.split(":", 1)[1].strip()
            # Clean up: take only the first word or handle parentheses
            # Example: "fear (threat response)" -> "fear"
            emotion_clean = emotion_raw.split("(")[0].strip().lower()
            result["emotion"] = emotion_clean
        elif line.startswith("- Speaker:") or line.startswith("Speaker:"):
            result["speaker"] = line.split(":", 1)[1].strip()
        elif line.startswith("- Impact:") or line.startswith("Impact:"):
            result["impact"] = line.split(":", 1)[1].strip()
    
    return result
