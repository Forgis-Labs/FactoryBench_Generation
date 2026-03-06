
# Templates for Pearl's Level 1: Observation (Seeing)

class ObservationTemplate:
    def __init__(self):
        self.templates = {
            "joint_pos": {
                "open": [
                    "Observing the telemetry, what is the current position of {joint}?",
                    "What is the angle of {joint} according to the latest sensor reading?",
                    "Check the telemetry: what is the value for {joint}?",
                    "Read the current angle of {joint} from the data stream."
                ],
                "mc": [
                    "Which of the following values matches the current angle of {joint}?",
                    "Select the correct position for {joint} from the options below.",
                    "Based on the telemetry, identify the angle of {joint}."
                ],
                "tf": [
                    "Is the angle of {joint} positive?",
                    "Is {joint} currently at a negative angle?",
                    "Does the telemetry show a value greater than 0 for {joint}?"
                ]
            },
            "op_mode": {
                "open": [
                    "What is the current operating mode of the {model}?",
                    "According to the system status, which mode is the {model} currently in?",
                    "Is the {model} in Manual, Auto, or Teach mode right now?"
                ],
                "mc": [
                    "Which operating mode is the {model} currently in?",
                    "Select the active mode for the {model} from the list.",
                    "Identify the current system status mode for the {model}."
                ],
                "tf": [
                    "Is the {model} currently in {target_mode} mode?",
                    "Does the system status indicate that the {model} is in {target_mode} mode?",
                    "Check the register: is the active mode {target_mode}?"
                ]
            },
            "error_status": {
                "open": [
                    "What warning is currently active in the status log?",
                    "Does the {model} have any active error codes defined in the logs?",
                    "Check the error register: is there an active warning?"
                ],
                "mc": [
                    "Which warning is currently logged in the status register?",
                    "Select the active error code from the options below.",
                    "Identify the warning present in the {model}'s logs."
                ],
                "tf": [
                    "Is there currently an active warning in the status log?",
                    "Does the log show any error for the {model}?",
                    "Is the error register clear of warnings?"
                ]
            }
        }

    def get_templates(self):
        return self.templates
