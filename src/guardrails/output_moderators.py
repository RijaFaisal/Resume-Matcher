import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModerationAction(Enum):
    """Actions to take based on moderation results."""

    ALLOW = "allow"
    WARN = "warn"
    FILTER = "filter"
    BLOCK = "block"


@dataclass
class ModerationResult:
    """Result of output moderation."""

    action: ModerationAction
    is_safe: bool
    violations: List[str]
    filtered_output: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.metadata is None:
            self.metadata = {}


class ToxicityFilter:
    """
    Detects toxic, harmful, or inappropriate content in output.

    Categories:
    - Profanity and offensive language
    - Hate speech
    - Sexual content
    - Violence and threats
    - Discrimination
    """

    def __init__(self, toxicity_threshold: float = 0.7):
        """
        Args:
            toxicity_threshold: Threshold for blocking (0.0-1.0)
        """
        self.toxicity_threshold = toxicity_threshold

        # Keyword-based toxicity detection (lightweight approach)
        # In production, use ML models like Perspective API or Detoxify
        self.toxic_keywords = {
            "profanity": [
                "damn",
                "hell",
                "crap",
                "shit",
                "fuck",
                "ass",
                "bitch",
                "bastard",
                "piss",
                "dick",
                "cock",
                "pussy",
                "cunt",
            ],
            "hate_speech": [
                "hate",
                "racist",
                "nazi",
                "supremacist",
                "kill all",
                "exterminate",
                "subhuman",
                "inferior race",
            ],
            "sexual": [
                "porn",
                "xxx",
                "nude",
                "naked",
                "sex tape",
                "erotic",
                "orgasm",
                "masturbat",
            ],
            "violence": [
                "kill",
                "murder",
                "shoot",
                "stab",
                "torture",
                "assault",
                "bomb",
                "weapon",
                "suicide",
                "self-harm",
            ],
            "discrimination": [
                "discriminate",
                "prejudice",
                "bigot",
                "stereotype",
                "sexist",
                "ageist",
                "homophobic",
            ],
        }

    def detect(self, text: str) -> ModerationResult:
        """
        Detect toxic content in output text.

        Args:
            text: Output text to moderate

        Returns:
            ModerationResult with toxicity assessment
        """
        violations = []
        category_scores = {}
        text_lower = text.lower()

        # Check each toxicity category
        for category, keywords in self.toxic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Calculate normalized score
                score = min(matches / 3.0, 1.0)  # 3+ matches = 1.0 score
                category_scores[category] = score
                violations.append(f"Detected {category}: {matches} keyword(s)")

        # Overall toxicity score (max of all categories)
        toxicity_score = max(category_scores.values()) if category_scores else 0.0

        # Determine action
        if toxicity_score >= self.toxicity_threshold:
            action = ModerationAction.BLOCK
            is_safe = False
        elif toxicity_score >= 0.4:
            action = ModerationAction.WARN
            is_safe = False
        else:
            action = ModerationAction.ALLOW
            is_safe = True

        logger.info(
            f"Toxicity Detection: score={toxicity_score:.2f}, action={action.value}"
        )

        return ModerationResult(
            action=action,
            is_safe=is_safe,
            violations=violations,
            confidence_scores={"toxicity": toxicity_score, **category_scores},
            metadata={"threshold": self.toxicity_threshold},
        )


class HallucinationDetector:
    """
    Detects potential hallucinations in LLM outputs.

    Checks for:
    - Uncertainty markers
    - Conflicting statements
    - Lack of specificity
    - Unsupported claims
    - Out-of-context information
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: Minimum confidence to consider output reliable
        """
        self.confidence_threshold = confidence_threshold

        # Patterns indicating uncertainty or hallucination
        self.uncertainty_patterns = [
            r"(?i)\b(might|may|could|possibly|perhaps|maybe|probably|likely)\b",
            r"(?i)\b(i think|i believe|i guess|i assume|in my opinion)\b",
            r"(?i)\b(not sure|uncertain|unclear|difficult to say)\b",
            r"(?i)\b(i don't know|i'm not certain|cannot confirm)\b",
        ]

        self.hallucination_indicators = [
            r"(?i)\b(as far as i know|to the best of my knowledge)\b",
            r"(?i)\b(without access to|i don\'t have information|i cannot verify)\b",
            r"(?i)\b(based on|according to)\s+(?!the provided|the context)",
        ]

    def detect(self, output: str, context: Optional[str] = None) -> ModerationResult:
        """
        Detect potential hallucinations in output.

        Args:
            output: Generated output text
            context: Original context/documents used (for grounding check)

        Returns:
            ModerationResult with hallucination assessment
        """
        violations = []
        confidence_score = 1.0

        # Check for uncertainty markers
        uncertainty_count = sum(
            len(re.findall(pattern, output)) for pattern in self.uncertainty_patterns
        )
        if uncertainty_count > 0:
            violations.append(f"Detected {uncertainty_count} uncertainty marker(s)")
            confidence_score -= uncertainty_count * 0.1

        # Check for hallucination indicators
        hallucination_count = sum(
            len(re.findall(pattern, output))
            for pattern in self.hallucination_indicators
        )
        if hallucination_count > 0:
            violations.append(
                f"Detected {hallucination_count} hallucination indicator(s)"
            )
            confidence_score -= hallucination_count * 0.15

        # Check for conflicting statements
        if self._has_contradictions(output):
            violations.append("Detected potential contradictions in output")
            confidence_score -= 0.2

        # Context grounding check
        if context:
            grounding_score = self._check_grounding(output, context)
            if grounding_score < 0.5:
                violations.append(f"Low grounding score: {grounding_score:.2f}")
                confidence_score = min(confidence_score, grounding_score)

        # Normalize confidence score
        confidence_score = max(0.0, min(1.0, confidence_score))

        # Determine action
        if confidence_score < 0.3:
            action = ModerationAction.BLOCK
            is_safe = False
        elif confidence_score < self.confidence_threshold:
            action = ModerationAction.WARN
            is_safe = False
        else:
            action = ModerationAction.ALLOW
            is_safe = True

        logger.info(
            f"Hallucination Detection: confidence={confidence_score:.2f}, action={action.value}"
        )

        return ModerationResult(
            action=action,
            is_safe=is_safe,
            violations=violations,
            confidence_scores={"confidence": confidence_score},
            metadata={
                "threshold": self.confidence_threshold,
                "uncertainty_count": uncertainty_count,
                "hallucination_indicators": hallucination_count,
            },
        )

    def _has_contradictions(self, text: str) -> bool:
        """Simple contradiction detection."""
        # Look for contradictory phrases
        contradiction_patterns = [
            (r"\byes\b", r"\bno\b"),
            (r"\btrue\b", r"\bfalse\b"),
            (r"\bis\b", r"\bis not\b"),
            (r"\bcan\b", r"\bcannot\b"),
            (r"\bwill\b", r"\bwill not\b"),
        ]

        text_lower = text.lower()
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, text_lower) and re.search(pattern2, text_lower):
                return True
        return False

    def _check_grounding(self, output: str, context: str) -> float:
        """
        Check if output is grounded in provided context.
        Returns grounding score (0.0-1.0).
        """
        # Simple keyword overlap approach
        # In production, use semantic similarity with embeddings
        output_words = set(re.findall(r"\b\w+\b", output.lower()))
        context_words = set(re.findall(r"\b\w+\b", context.lower()))

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "are",
            "was",
            "were",
        }
        output_words -= stop_words
        context_words -= stop_words

        if not output_words:
            return 0.0

        overlap = len(output_words & context_words)
        grounding_score = overlap / len(output_words)

        return grounding_score


class OutputModerator:
    """
    Comprehensive output moderator combining multiple checks.
    """

    def __init__(
        self,
        enable_toxicity_filter: bool = True,
        enable_hallucination_detector: bool = True,
        toxicity_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            enable_toxicity_filter: Enable toxicity filtering
            enable_hallucination_detector: Enable hallucination detection
            toxicity_threshold: Threshold for toxicity blocking
            confidence_threshold: Minimum confidence for output
        """
        self.enable_toxicity_filter = enable_toxicity_filter
        self.enable_hallucination_detector = enable_hallucination_detector

        if enable_toxicity_filter:
            self.toxicity_filter = ToxicityFilter(toxicity_threshold=toxicity_threshold)

        if enable_hallucination_detector:
            self.hallucination_detector = HallucinationDetector(
                confidence_threshold=confidence_threshold
            )

    def moderate(self, output: str, context: Optional[str] = None) -> ModerationResult:
        """
        Run all enabled moderation checks on output.

        Args:
            output: Generated output to moderate
            context: Original context used for generation

        Returns:
            ModerationResult combining all checks
        """
        all_violations = []
        all_scores = {}
        highest_action = ModerationAction.ALLOW

        # Toxicity check
        if self.enable_toxicity_filter:
            toxicity_result = self.toxicity_filter.detect(output)
            all_violations.extend(toxicity_result.violations)
            all_scores.update(toxicity_result.confidence_scores)
            if toxicity_result.action.value != "allow":
                highest_action = max(
                    highest_action,
                    toxicity_result.action,
                    key=lambda x: list(ModerationAction).index(x),
                )

        # Hallucination check
        if self.enable_hallucination_detector:
            hallucination_result = self.hallucination_detector.detect(output, context)
            all_violations.extend(hallucination_result.violations)
            all_scores.update(hallucination_result.confidence_scores)
            if hallucination_result.action.value != "allow":
                highest_action = max(
                    highest_action,
                    hallucination_result.action,
                    key=lambda x: list(ModerationAction).index(x),
                )

        # Overall safety
        is_safe = highest_action in [ModerationAction.ALLOW, ModerationAction.WARN]

        # Create filtered output if needed
        filtered_output = None
        if highest_action == ModerationAction.BLOCK:
            filtered_output = "[Output blocked due to policy violations]"
        elif highest_action == ModerationAction.FILTER:
            filtered_output = "[Output filtered - some content removed]"

        return ModerationResult(
            action=highest_action,
            is_safe=is_safe,
            violations=all_violations,
            filtered_output=filtered_output,
            confidence_scores=all_scores,
            metadata={
                "checks_run": [
                    "toxicity" if self.enable_toxicity_filter else None,
                    "hallucination" if self.enable_hallucination_detector else None,
                ]
            },
        )
