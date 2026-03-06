
import random
from typing import Dict, Any, List, Optional

class TelemetryTemplate:
    def __init__(self):
        # Templates for statistical queries (Level 1)
        self.stats_templates = {
            "mean": [
                "What is the average value of the signal?",
                "Calculate the mean of the observed values.",
                "What is the mean value over the given period?"
            ],
            "max": [
                "What is the maximum value reached?",
                "Identify the peak value in the sequence.",
                "What is the highest recorded value?"
            ],
            "min": [
                "What is the minimum value observed?",
                "Identify the lowest value in the sequence.",
                "What is the lowest recorded value?"
            ],
            "std": [
                "What is the standard deviation of the signal?",
                "How much does the signal vary from the mean (std)?",
                "Calculate the standard deviation."
            ]
        }

        # Templates for pattern recognition (Level 2)
        self.pattern_templates = [
            "What type of pattern is shown in the data?",
            "Classify the shape of this signal.",
            "Describe the behavior of the signal over time."
        ]

        # Templates for identifying specific values at timestamps (Level 1)
        self.point_templates = [
            "What is the value at timestamp {t}?",
            "Report the signal value at t={t}.",
            "At time {t}, what was the recorded value?"
        ]

    def get_stats_question(self, metric: str, value: float, format: str = "open") -> Dict[str, Any]:
        """Generates a question about a specific statistical metric."""
        if metric not in self.stats_templates:
            return None
            
        template = random.choice(self.stats_templates[metric])
        question_text = template
        answer_value = str(value)
        options = []
        
        if format == "mc":
            # Generate distractors
            distractors = [
                value * random.uniform(0.8, 1.2),
                value + random.uniform(-10, 10),
                value * random.uniform(0.5, 1.5)
            ]
            # Ensure unique and sorted options
            options = sorted(list(set([value] + distractors)), key=lambda x: random.random())
            options = [round(opt, 6) for opt in options] 
            question_text += " Choose the closest value."
            
        elif format == "tf":
            is_true = random.choice([True, False])
            if is_true:
                threshold = value
                direction = "equal to"
            else:
                threshold = value * random.uniform(0.8, 1.2)
                if threshold == value: threshold += 0.1
                direction = "equal to"
                
            question_text = f"Is the {metric} of the signal {direction} {threshold:.6f}?"
            answer_value = str(is_true)
        
        result = {
            "question": {
                "text": question_text,
                "type": f"level1_stats_{metric}_{format}",
                "pearl_layer": "Associational"
            },
            "answer": {
                "value": answer_value,
                "confidence": 1.0,
                "type": "numeric" if format != "tf" else "boolean"
            }
        }
        
        if options:
            result["question"]["options"] = options
            
        return result

    def get_pattern_question(self, subtype: str) -> Dict[str, Any]:
        """Generates a question about the signal pattern/subtype."""
        template = random.choice(self.pattern_templates)
        
        return {
            "question": {
                "text": template,
                "type": "level2_pattern_recognition",
                "pearl_layer": "Associational" # border of Interventional if it implies mechanism
            },
            "answer": {
                "value": subtype,
                "confidence": 1.0,
                "type": "categorical"
            }
        }
    
    def get_point_question(self, timestamp: float, value: float, format: str = "open") -> Dict[str, Any]:
        """Generates a question about a value at a specific timestamp."""
        template = random.choice(self.point_templates)
        text = template.format(t=timestamp)
        question_text = text
        answer_value = str(value)
        options = []
        
        if format == "mc":
             # Generate distractors
            distractors = [
                value * random.uniform(0.9, 1.1),
                value + random.uniform(-5, 5),
                value * -1
            ]
            options = sorted(list(set([value] + distractors)), key=lambda x: random.random())
            options = [round(opt, 6) for opt in options]
            question_text += " Select the correct value."
            
        elif format == "tf":
            is_true = random.choice([True, False])
            threshold = value if is_true else value + random.choice([-1, 1]) * random.uniform(1, 10)
            question_text = f"At time {timestamp}, is the value {threshold:.6f}?"
            answer_value = str(is_true)
            
        result = {
            "question": {
                "text": question_text,
                "type": f"level1_point_query_{format}",
                "pearl_layer": "Associational"
            },
            "answer": {
                "value": answer_value,
                "confidence": 1.0,
                "type": "numeric" if format != "tf" else "boolean"
            }
        }
        
        if options:
            result["question"]["options"] = options

        return result

