class DetectionResult:
    def __init__(self, cls, confidence, result_number, color, aspect_ratio, class_name, original_class_name):
        self.cls = cls
        self.confidence = confidence
        self.result_number = result_number
        self.color = color
        self.aspect_ratio = aspect_ratio
        self.class_name = class_name
        self.original_class_name = original_class_name

    def __repr__(self):
        return f"DetectionResult(class={self.cls}, confidence={self.confidence}, result_number={self.result_number}, color={self.color}, class_name={self.class_name}, original_class_name={self.original_class_name})"

    def to_dict(self):
        return {
            "class": self.cls,
            "class_name": self.class_name,
            "original_class_name": self.original_class_name,
            "confidence": self.confidence,
            "result_number": self.result_number,
            "color": self.color,
            "aspect_ratio": self.aspect_ratio
        }