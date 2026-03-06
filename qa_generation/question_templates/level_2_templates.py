import random
import yaml
import os
import math
from typing import Dict, Any, List, Optional, Union

# Try to import generated ranges, fallback to default if not found
try:
    from utils.operating_ranges_from_manuals import OPERATING_RANGES
except ImportError:
    OPERATING_RANGES = {}

class Level2Template:
    def __init__(self, prompts_path: str = "prompts/level_2_prompts.yaml"):
        self.prompts = self._load_prompts(prompts_path)
        self.operating_ranges = OPERATING_RANGES
        
    def _load_prompts(self, path: str) -> Dict:
        if not os.path.exists(path):
            path = os.path.join("prompts", "level_2_prompts.yaml")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _fill_template(self, template: str, context: Dict[str, Any]) -> str:
        try:
            return template.format(**context)
        except Exception:
            return template

    def get_robot_specs(self, robot_model: str) -> Dict:
        # Match robot model to keys in OPERATING_RANGES
        for key in self.operating_ranges:
            if key in robot_model or robot_model in key:
                return self.operating_ranges[key]
        return {}

    def get_anomaly_question(self, robot_model: str, episode_data: Optional[Dict], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates anomaly question with DETECTED anomaly based on context/data.
        Context must contain 'anomaly_present' boolean or we derive it.
        """
        category = self.prompts.get('anomaly_detection', {})
        q_type = random.choice(['binary', 'open_ended', 'multiple_choice'])
        
        # Determine finding
        anomaly_present = context.get("anomaly_present", False)
        anomaly_type = context.get("anomaly_type", "None")
        joint_name = context.get("joint_name", "joint_1")
        
        # If binary
        if q_type == 'binary':
            templates = category.get('binary', [])
            if not templates: return None
            template = random.choice(templates)
            question_text = self._fill_template(template, context)
            
            # Deterministic Answer
            if "anomaly" in question_text.lower() or "abnormal" in question_text.lower():
                answer_val = "True" if anomaly_present else "False"
            elif "normal" in question_text.lower():
                answer_val = "False" if anomaly_present else "True"
            else:
                answer_val = "Yes" if anomaly_present else "No" # Default fallback
                
            return {
                "question": {
                    "text": question_text,
                    "type": f"level2_anomaly_binary",
                    "pearl_layer": "Interventional"
                },
                "answer": {
                    "value": answer_val,
                    "confidence": 1.0,
                    "type": "boolean"
                }
            }
            
        # If open ended
        elif q_type == 'open_ended':
            templates = category.get('open_ended', [])
            if not templates: return None
            template = random.choice(templates)
            question_text = self._fill_template(template, context)
            
            if anomaly_present:
                answer_val = f"Detected {anomaly_type} in {joint_name}. Values exceeded safety limits."
            else:
                answer_val = "No anomalies detected. All parameters are within normal operating ranges."
                
            return {
                "question": {
                    "text": question_text,
                    "type": f"level2_anomaly_open",
                    "pearl_layer": "Interventional"
                },
                "answer": {
                    "value": answer_val,
                    "confidence": 1.0,
                    "type": "text"
                }
            }

        # If multiple choice
        elif q_type == 'multiple_choice':
            templates = category.get('multiple_choice', [])
            if not templates: return None
            template_obj = random.choice(templates)
            question_text = self._fill_template(template_obj['template'], context)
            options = [self._fill_template(opt, context) for opt in template_obj['options']]
            
            # Answer logic
            if anomaly_present:
                 # Try to find the anomaly type in options
                 answer_val = next((opt for opt in options if anomaly_type.lower() in opt.lower()), options[-1])
                 # If checking for joint name
                 if "joint" in question_text.lower():
                     answer_val = next((opt for opt in options if joint_name in opt), "None")
            else:
                answer_val = "None" or "No anomaly"
                # Find option resembling "None"
                for opt in options:
                    if "no " in opt.lower() or "none" in opt.lower():
                        answer_val = opt
                        break

            return {
                "question": {
                    "text": question_text,
                    "type": f"level2_anomaly_mc",
                    "options": options,
                    "pearl_layer": "Interventional"
                },
                "answer": {
                    "value": answer_val,
                    "confidence": 1.0,
                    "type": "categorical"
                }
            }

    def get_intervention_question(self, robot_model: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates what-if using basic physics models.
        """
        category = self.prompts.get('intervention', {})
        q_type = "what_if" # Focus on what_if for physics
        templates = category.get(q_type, [])
        if not templates: return None
        template = random.choice(templates)
        
        # We need specific templates to apply specific physics
        # Heuristic: match template keywords to physics logic
        
        specs = self.get_robot_specs(robot_model)
        
        question_text = self._fill_template(template, context)
        answer_val = "Simulation required."
        
        # Physics Logic 1: Speed vs Temperature (Kinetic Energy)
        if "temperature" in question_text.lower() and "speed" in question_text.lower():
            # T_new approx T_old + k * (v_new^2 - v_old^2)
            try:
                current_speed = float(context.get("speed", 0))
                # extracting new speed from text or context?? 
                # context should have 'rpm' or similar if used in template
                # simplified: assume context has 'new_value' and 'current_value' mapped correctly
                v1 = float(context.get("current_value", 0))
                v2 = float(context.get("new_value", v1*1.5))
                t1 = float(context.get("output_variable_current", 40)) # current temp
                
                # Dummy coefficient
                k = 0.001 
                t2 = t1 + k * (v2**2 - v1**2)
                
                limit = specs.get("temp_range", {}).get("max", 50)
                safety = "within limits" if t2 < limit else "exceeding safety limits"
                
                answer_val = f"The temperature would rise to approximately {t2:.1f}C, which is {safety}."
            except:
                answer_val = "Temperature would likely increase with speed."

        # Physics Logic 2: Load vs Joint Limits
        elif "load" in question_text.lower() and "weight" in question_text.lower():
            try:
                load = float(context.get("weight", 0))
                # Check against payload if available (std 20kg for IRB 2600)
                max_load = 20.0 # Default
                
                if load > max_load:
                    answer_val = f"With {load}kg, the robot would exceed its rated payload of {max_load}kg, likely causing motor stall or safety stop."
                else:
                    answer_val = f"The load of {load}kg is within the rated capacity."
            except:
                 answer_val = "Checking payload limits."
        
        # Fallback for other intervention types if needed...

        return {
            "question": {
                "text": question_text,
                "type": f"level2_intervention_physics",
                "pearl_layer": "Interventional"
            },
            "answer": {
                "value": answer_val,
                "confidence": 0.85,
                "type": "text"
            }
        }
