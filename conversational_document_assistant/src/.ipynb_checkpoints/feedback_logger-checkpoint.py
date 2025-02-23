import json

class FeedbackLogger:
    def __init__(self, log_file="feedback_log.json"):
        self.log_file = log_file

    def log_feedback(self, user_input, assistant_response, feedback):
        log_entry = {
            "user_input": user_input,
            "assistant_response": assistant_response,
            "feedback": feedback
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
