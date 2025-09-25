import time
import random
import string
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.PredictionSamplesData import PredictionSamplesData
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger


def generate_random_word(min_length=3, max_length=12):
    """Generate a random word with random length"""
    length = random.randint(min_length, max_length)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def generate_random_text(num_words=None):
    """Generate random text with random number of words"""
    if num_words is None:
        num_words = random.randint(3, 15)
    return " ".join(generate_random_word() for _ in range(num_words))


def generate_random_identifier():
    """Generate random alphanumeric identifier"""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(6, 15)))


def generate_realistic_training_samples(num_samples: int) -> list[TrainingSample]:
    """Generate completely random training samples for regex subtraction testing"""
    prefix_patterns = ["{}: ", "{} ", "{}= ", "{}-", "{}# ", "{} Number: ", "{} ID: ", "{} Code: ", "{} Ref: "]
    suffix_patterns = [" - {}", " ({})", " [{}]", " - {}", " STATUS: {}", " - {}", " {}", " - {}", " ({})", " - {}"]

    samples = []

    for i in range(num_samples):
        # Generate random content to extract
        content = generate_random_identifier()

        # Random pattern type
        pattern_type = random.choice(["prefix", "suffix", "both", "neither", "multiple"])

        if pattern_type == "prefix":
            prefix_word = generate_random_word()
            prefix = random.choice(prefix_patterns).format(prefix_word)
            full_text = f"{prefix}{content}"
        elif pattern_type == "suffix":
            suffix_word = generate_random_word()
            suffix = random.choice(suffix_patterns).format(suffix_word)
            full_text = f"{content}{suffix}"
        elif pattern_type == "both":
            prefix_word = generate_random_word()
            suffix_word = generate_random_word()
            prefix = random.choice(prefix_patterns).format(prefix_word)
            suffix = random.choice(suffix_patterns).format(suffix_word)
            full_text = f"{prefix}{content}{suffix}"
        elif pattern_type == "multiple":
            # Add multiple instances of the content with different patterns
            instances = []
            for _ in range(random.randint(2, 4)):
                instance_content = content if random.random() > 0.3 else generate_random_identifier()
                prefix_word = generate_random_word()
                prefix = random.choice(prefix_patterns).format(prefix_word)
                instances.append(f"{prefix}{instance_content}")
            full_text = " ".join(instances)
        else:  # neither - just the content with random text
            random_before = generate_random_text(random.randint(1, 5))
            random_after = generate_random_text(random.randint(1, 5))
            full_text = f"{random_before} {content} {random_after}"

        sample = TrainingSample(labeled_data=LabeledData(label_text=content, language_iso="en", source_text=full_text))
        samples.append(sample)

    return samples


def load_test_regex_subtraction_method():
    """Load test for RegexSubtractionMethod with realistic data"""
    print("Starting load test for RegexSubtractionMethod...")

    extraction_identifier = ExtractionIdentifier(run_name="load_test", extraction_name="regex_subtraction_test")
    logger = ExtractorLogger()

    # Generate training samples
    num_training_samples = 1000
    print(f"Generating {num_training_samples} training samples...")
    training_samples = generate_realistic_training_samples(num_training_samples)

    extraction_data = ExtractionData(samples=training_samples, extraction_identifier=extraction_identifier)

    # Initialize method
    regex_method = RegexSubtractionMethod(extraction_identifier, logger)

    # Test training
    print("Training RegexSubtractionMethod...")
    start_time = time.time()
    regex_method.train(extraction_data)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Test performance evaluation
    print("Evaluating performance...")
    start_time = time.time()
    performance = regex_method.get_performance(extraction_data, extraction_data)
    performance_time = time.time() - start_time
    print(f"Performance evaluation completed in {performance_time:.2f} seconds")
    print(f"Performance score: {performance}")

    # Test prediction
    print("Testing predictions...")
    test_samples = generate_realistic_training_samples(100)
    prediction_samples_data = PredictionSamplesData(
        prediction_samples=[PredictionSample(source_text=sample.labeled_data.source_text) for sample in test_samples],
        options=[],
        multi_value=False,
    )

    start_time = time.time()
    predictions = regex_method.predict(prediction_samples_data)
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Generated {len(predictions)} predictions")

    print("Load test completed successfully!")
    return {
        "training_time": training_time,
        "performance_time": performance_time,
        "prediction_time": prediction_time,
        "performance_score": performance,
        "num_predictions": len(predictions),
    }


if __name__ == "__main__":
    load_test_regex_subtraction_method()
