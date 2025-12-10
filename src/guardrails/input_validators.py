import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for validation results."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    risk_level: RiskLevel
    violations: List[str]
    sanitized_input: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PIIDetector:
    """
    Detects Personally Identifiable Information (PII) in text.

    Patterns detected:
    - Email addresses
    - Phone numbers (US, international formats)
    - Social Security Numbers (SSN)
    - Credit card numbers
    - IP addresses
    - Physical addresses
    """

    def __init__(self, mask_pii: bool = True):
        """
        Args:
            mask_pii: If True, replace detected PII with placeholders
        """
        self.mask_pii = mask_pii

        # Regex patterns for PII detection
        self.patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone_us": r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b",
            "phone_intl": r"\+[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}",
            "ssn": r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "address": r"\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir)\b",
        }

    def detect(self, text: str) -> ValidationResult:
        """
        Detect PII in the input text.

        Args:
            text: Input text to scan for PII

        Returns:
            ValidationResult with detected PII types
        """
        violations = []
        pii_found = {}
        sanitized = text

        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            found = list(matches)

            if found:
                violations.append(f"Detected {pii_type}: {len(found)} instance(s)")
                pii_found[pii_type] = len(found)

                if self.mask_pii:
                    # Replace with placeholder
                    placeholder = f"[{pii_type.upper()}_REDACTED]"
                    sanitized = re.sub(
                        pattern, placeholder, sanitized, flags=re.IGNORECASE
                    )

        # Determine risk level
        if not violations:
            risk_level = RiskLevel.SAFE
            is_valid = True
        elif len(violations) <= 2:
            risk_level = RiskLevel.LOW
            is_valid = True  # Allow with warning
        elif len(violations) <= 4:
            risk_level = RiskLevel.MEDIUM
            is_valid = False
        else:
            risk_level = RiskLevel.HIGH
            is_valid = False

        logger.info(
            f"PII Detection: {len(violations)} violation(s) found, risk={risk_level.value}"
        )

        return ValidationResult(
            is_valid=is_valid,
            risk_level=risk_level,
            violations=violations,
            sanitized_input=sanitized if self.mask_pii else None,
            metadata={"pii_counts": pii_found},
        )


class PromptInjectionFilter:
    """
    Detects and filters potential prompt injection attacks, SQL injection, and XSS attacks.

    Checks for:
    - System prompt overrides
    - Role manipulation
    - Instruction hijacking
    - Delimiter confusion
    - Encoding tricks
    - SQL injection attempts
    - Cross-Site Scripting (XSS) attempts
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, applies more aggressive filtering
        """
        self.strict_mode = strict_mode

        # Patterns indicating prompt injection attempts
        self.injection_patterns = [
            # System role manipulation
            r"(?i)(ignore|disregard|forget)\s+(previous|prior|all|above|system)\s+(instructions?|prompts?|rules?)",
            r"(?i)(you are|act as|pretend to be|roleplay as)\s+(now |a )?(?:system|admin|root|god mode)",
            # Instruction override
            r"(?i)##\s*new\s+(instructions?|task|system prompt)",
            r"(?i)(override|replace|update)\s+(your|the)\s+(instructions?|rules?|system prompt)",
            # Delimiter confusion
            r'(```|"""|\'\'\')\s*system',
            r"<\|system\|>|<\|end\|>|<\|im_start\|>",
            # Direct commands
            r"(?i)^(system:|assistant:|user:)",
            r"(?i)\[SYSTEM\]|\[INST\]|\[/INST\]",
            # Encoding tricks
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
            r"&#[0-9]{1,4};",  # HTML entities
            # Jailbreak attempts
            r"(?i)(DAN|based mode|developer mode|jailbreak)",
            r"(?i)(sudo|root|admin)\s+(mode|access|privileges)",
            # SQL Injection patterns
            r"(?i)('\s*or\s*'1'\s*=\s*'1|1\s*=\s*1)",  # Classic SQL injection
            r"(?i)'\s*or\s*1\s*=\s*1\s*(--|#|/\*)",  # SQL injection with comments
            r"(?i)(union\s+select|union\s+all\s+select)",  # UNION-based injection
            r"(?i)(insert\s+into|update\s+.+\s+set|delete\s+from)",  # DML statements
            r"(?i)(drop\s+table|drop\s+database|truncate\s+table)",  # DDL statements
            r"(?i)(exec(\s|\()|execute(\s|\())",  # Command execution
            r"(?i)(xp_cmdshell|sp_executesql)",  # SQL Server specific
            r"--\s*$|;\s*--|#\s*$",  # SQL comments at end of line
            r"(?i)(\bor\b|\band\b).*'.*=.*'",  # Boolean-based blind injection
            r"(?i)(sleep\(|benchmark\(|waitfor\s+delay)",  # Time-based blind injection
            # XSS (Cross-Site Scripting) patterns
            r"(?i)<script[^>]*>.*?</script>",  # Script tags
            r"(?i)<script[^>]*>",  # Opening script tag
            r"(?i)on(error|load|click|mouse|focus|blur|change|submit|key)\s*=",  # Event handlers
            r"(?i)javascript:",  # JavaScript protocol
            r"(?i)<iframe[^>]*>",  # Iframe injection
            r"(?i)<object[^>]*>",  # Object tag
            r"(?i)<embed[^>]*>",  # Embed tag
            r"(?i)<img[^>]*on\w+\s*=",  # Image with event handler
            r"(?i)eval\s*\(",  # JavaScript eval
            r"(?i)expression\s*\(",  # CSS expression
            r"(?i)<svg[^>]*onload",  # SVG with onload
            r"(?i)alert\s*\(",  # Alert function (common in XSS tests)
            r"(?i)document\.cookie",  # Cookie stealing
            r"(?i)document\.write",  # DOM manipulation
            r"(?i)<base[^>]*>",  # Base tag injection
        ]

        # Additional strict mode patterns
        if self.strict_mode:
            self.injection_patterns.extend(
                [
                    r"(?i)(tell me|show me|reveal|expose)\s+(your|the)\s+(prompt|instructions|system)",
                    r"(?i)translate to (python|javascript|code)",
                    r"(?i)<\w+[^>]*style\s*=",  # Style attribute (potential XSS)
                    r"(?i)data:text/html",  # Data URI XSS
                ]
            )

    def detect(self, text: str) -> ValidationResult:
        """
        Detect potential prompt injection, SQL injection, and XSS attempts.

        Args:
            text: Input text to check for injection patterns

        Returns:
            ValidationResult with detected injection attempts
        """
        violations = []
        detected_patterns = []
        attack_types = {"prompt_injection": 0, "sql_injection": 0, "xss": 0, "other": 0}

        # Categorize patterns for better reporting
        prompt_keywords = ["ignore", "disregard", "system", "admin", "jailbreak", "DAN"]
        sql_keywords = [
            "union",
            "select",
            "drop",
            "insert",
            "delete",
            "exec",
            "waitfor",
            "--",
            "or.*=",
        ]
        xss_keywords = [
            "script",
            "onerror",
            "onload",
            "javascript:",
            "iframe",
            "eval",
            "alert",
        ]

        for pattern in self.injection_patterns:
            matches = re.finditer(pattern, text)
            found = list(matches)

            if found:
                # Categorize the attack type
                attack_type = "other"
                if any(kw in pattern.lower() for kw in sql_keywords):
                    attack_type = "sql_injection"
                    violations.append("SQL injection pattern detected")
                    attack_types["sql_injection"] += 1
                elif any(kw in pattern.lower() for kw in xss_keywords):
                    attack_type = "xss"
                    violations.append("XSS pattern detected")
                    attack_types["xss"] += 1
                elif any(kw in pattern.lower() for kw in prompt_keywords):
                    attack_type = "prompt_injection"
                    violations.append("Prompt injection pattern detected")
                    attack_types["prompt_injection"] += 1
                else:
                    violations.append(f"Suspicious pattern detected: {pattern[:50]}...")
                    attack_types["other"] += 1

                detected_patterns.append(pattern)

        # Check for excessive special characters (possible encoding attack)
        special_char_ratio = len(re.findall(r"[^\w\s]", text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            violations.append(f"High special character ratio: {special_char_ratio:.2%}")
            attack_types["other"] += 1

        # Determine risk level based on attack types
        if not violations:
            risk_level = RiskLevel.SAFE
            is_valid = True
        elif attack_types["sql_injection"] > 0 or attack_types["xss"] > 0:
            # SQL and XSS are critical security threats
            risk_level = RiskLevel.CRITICAL
            is_valid = False
        elif len(violations) == 1:
            risk_level = RiskLevel.LOW
            is_valid = not self.strict_mode
        elif len(violations) <= 3:
            risk_level = RiskLevel.MEDIUM
            is_valid = False
        else:
            risk_level = RiskLevel.CRITICAL
            is_valid = False

        if violations:
            logger.warning(
                f"Security Check: {len(violations)} violation(s) detected, "
                f"risk={risk_level.value}, types={attack_types}"
            )
        else:
            logger.info("Security Check: No threats detected")

        return ValidationResult(
            is_valid=is_valid,
            risk_level=risk_level,
            violations=violations,
            metadata={
                "patterns_matched": len(detected_patterns),
                "attack_types": attack_types,
                "total_violations": len(violations),
            },
        )


class InputValidator:
    """
    Comprehensive input validator combining multiple checks.
    """

    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_injection_filter: bool = True,
        mask_pii: bool = True,
        strict_mode: bool = False,
        max_length: int = 50000,
        min_length: int = 10,
    ):
        """
        Args:
            enable_pii_detection: Enable PII detection
            enable_injection_filter: Enable prompt injection filtering
            mask_pii: Mask detected PII
            strict_mode: Use strict filtering rules
            max_length: Maximum allowed input length
            min_length: Minimum allowed input length
        """
        self.enable_pii_detection = enable_pii_detection
        self.enable_injection_filter = enable_injection_filter
        self.max_length = max_length
        self.min_length = min_length

        if enable_pii_detection:
            self.pii_detector = PIIDetector(mask_pii=mask_pii)

        if enable_injection_filter:
            self.injection_filter = PromptInjectionFilter(strict_mode=strict_mode)

    def validate(self, text: str) -> ValidationResult:
        """
        Run all enabled validation checks on input text.

        Args:
            text: Input text to validate

        Returns:
            ValidationResult combining all checks
        """
        all_violations = []
        highest_risk = RiskLevel.SAFE
        sanitized = text

        # Length validation
        if len(text) < self.min_length:
            all_violations.append(f"Input too short: {len(text)} < {self.min_length}")
            return ValidationResult(
                is_valid=False, risk_level=RiskLevel.LOW, violations=all_violations
            )

        if len(text) > self.max_length:
            all_violations.append(f"Input too long: {len(text)} > {self.max_length}")
            return ValidationResult(
                is_valid=False, risk_level=RiskLevel.MEDIUM, violations=all_violations
            )

        # PII detection
        if self.enable_pii_detection:
            pii_result = self.pii_detector.detect(text)
            all_violations.extend(pii_result.violations)
            if pii_result.risk_level.value != "safe":
                highest_risk = max(
                    highest_risk,
                    pii_result.risk_level,
                    key=lambda x: list(RiskLevel).index(x),
                )
            if pii_result.sanitized_input:
                sanitized = pii_result.sanitized_input

        # Prompt injection check
        if self.enable_injection_filter:
            injection_result = self.injection_filter.detect(text)
            all_violations.extend(injection_result.violations)
            if injection_result.risk_level.value != "safe":
                highest_risk = max(
                    highest_risk,
                    injection_result.risk_level,
                    key=lambda x: list(RiskLevel).index(x),
                )

        # Overall validity
        is_valid = highest_risk in [RiskLevel.SAFE, RiskLevel.LOW]

        return ValidationResult(
            is_valid=is_valid,
            risk_level=highest_risk,
            violations=all_violations,
            sanitized_input=sanitized if sanitized != text else None,
            metadata={
                "checks_run": [
                    "length",
                    "pii" if self.enable_pii_detection else None,
                    "injection" if self.enable_injection_filter else None,
                ],
                "original_length": len(text),
            },
        )
