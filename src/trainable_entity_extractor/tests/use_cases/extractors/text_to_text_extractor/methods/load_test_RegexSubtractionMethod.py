import time
import random
import string
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.text_to_text_extractor.methods.RegexSubtractionMethod import (
    RegexSubtractionMethod,
)


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
    # Random prefixes and suffixes with random words
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
            prefix = random.choice(prefix_patterns).get_text(prefix_word)
            full_text = f"{prefix}{content}"
            label_text = content
        elif pattern_type == "suffix":
            suffix_word = generate_random_word()
            suffix = random.choice(suffix_patterns).get_text(suffix_word)
            full_text = f"{content}{suffix}"
            label_text = content
        elif pattern_type == "both":
            prefix_word = generate_random_word()
            suffix_word = generate_random_word()
            prefix = random.choice(prefix_patterns).get_text(prefix_word)
            suffix = random.choice(suffix_patterns).get_text(suffix_word)
            full_text = f"{prefix}{content}{suffix}"
            label_text = content
        elif pattern_type == "multiple":
            # Multiple occurrences of the content with different patterns
            prefix_word = generate_random_word()
            suffix_word = generate_random_word()
            prefix = random.choice(prefix_patterns).get_text(prefix_word)
            suffix = random.choice(suffix_patterns).get_text(suffix_word)
            # Add the content multiple times in different contexts
            extra_content = generate_random_identifier()
            full_text = f"{generate_random_text(2)} {prefix}{extra_content} {generate_random_text(1)} {content}{suffix} {generate_random_text(2)}"
            label_text = content
        else:  # neither - just the content
            full_text = content
            label_text = content

        # Add completely random noise around the text
        noise_before = ""
        noise_after = ""

        if random.random() < 0.7:  # 70% chance of noise before
            noise_before = generate_random_text(random.randint(1, 8)) + ". "

        if random.random() < 0.7:  # 70% chance of noise after
            noise_after = ". " + generate_random_text(random.randint(1, 8))

        # Sometimes add line breaks and extra formatting
        if random.random() < 0.3:
            noise_before = noise_before + "\n" + generate_random_text(random.randint(1, 3)) + " "

        if random.random() < 0.3:
            noise_after = " \n" + generate_random_text(random.randint(1, 3)) + noise_after

        source_text = f"{noise_before}{full_text}{noise_after}".strip()

        labeled_data = LabeledData(
            label_text=label_text,
            source_text=source_text,
            entity_name=f"{generate_random_word()}_{i % 50}",  # More variety in entity names
            tenant=f"tenant_{generate_random_word()}",
            id=f"sample_{i}_{generate_random_word(4, 6)}",
        )

        training_sample = TrainingSample(labeled_data=labeled_data)
        samples.append(training_sample)

    return samples


def generate_realistic_prediction_samples(num_samples: int) -> list[PredictionSample]:
    """Generate completely random prediction samples"""
    samples = []

    for i in range(num_samples):
        content = generate_random_identifier()

        # Mix different random patterns
        pattern_type = random.choice(["prefix", "suffix", "both", "clean", "complex", "nested"])

        if pattern_type == "prefix":
            prefix_word = generate_random_word()
            prefix = random.choice([f"{prefix_word}: ", f"{prefix_word} ", f"{prefix_word}= "])
            text = f"{prefix}{content}"
        elif pattern_type == "suffix":
            suffix_word = generate_random_word()
            suffix = random.choice([f" - {suffix_word}", f" ({suffix_word})", f" [{suffix_word}]"])
            text = f"{content}{suffix}"
        elif pattern_type == "both":
            prefix_word = generate_random_word()
            suffix_word = generate_random_word()
            text = f"{prefix_word}: {content} - {suffix_word}"
        elif pattern_type == "complex":
            # More complex nested patterns
            outer_prefix = generate_random_word()
            inner_prefix = generate_random_word()
            suffix = generate_random_word()
            text = f"{outer_prefix} {inner_prefix}: {content} ({suffix})"
        elif pattern_type == "nested":
            # Nested patterns with multiple identifiers
            other_content = generate_random_identifier()
            middle_word = generate_random_word()
            text = f"{generate_random_word()}: {other_content} {middle_word} {content} {generate_random_word()}"
        else:  # clean
            text = content

        # Add random contextual noise with more variety
        noise_options = [
            "",
            generate_random_text(random.randint(1, 5)) + ". ",
            f"{generate_random_word()} {generate_random_word()}: ",
            f"{generate_random_word()} {i}: ",
            generate_random_text(random.randint(2, 6)) + "\n",
            f"[{generate_random_word()}] ",
            f"#{random.randint(1000, 9999)} - ",
        ]

        noise = random.choice(noise_options)

        # Sometimes add trailing noise
        if random.random() < 0.4:
            trailing_noise = random.choice(
                [
                    f" - {generate_random_word()}",
                    f"\n{generate_random_text(random.randint(1, 4))}",
                    f" ({generate_random_word()})",
                    f" | {generate_random_word()}: {generate_random_word()}",
                ]
            )
            text += trailing_noise

        full_text = f"{noise}{text}".strip()
        sample = PredictionSample.from_text(full_text)
        samples.append(sample)

    return samples


def test_regex_subtraction_method_comprehensive_load():
    """Comprehensive load test with completely random training and prediction data"""
    print("Starting comprehensive RegexSubtractionMethod load test with random data...")

    # Setup
    identifier = ExtractionIdentifier(run_name="random_load_test", extraction_name="regex_random_load")
    method = RegexSubtractionMethod(extraction_identifier=identifier)

    # Test parameters - much larger for stress testing
    num_training_samples = 20000
    num_prediction_samples = 50000

    print(f"Generating {num_training_samples} completely random training samples...")
    start_gen = time.time()
    training_samples = generate_realistic_training_samples(num_training_samples)
    end_gen = time.time()
    print(f"Training samples generated in {end_gen - start_gen:.2f} seconds")

    # Create extraction data
    extraction_data = ExtractionData(samples=training_samples, extraction_identifier=identifier)

    # Training phase
    print(f"Training with {num_training_samples} random samples...")
    start_train = time.time()
    method.train(extraction_data)
    end_train = time.time()
    training_time = end_train - start_train
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training throughput: {num_training_samples / training_time:.2f} samples/second")

    # Generate prediction samples
    print(f"Generating {num_prediction_samples} completely random prediction samples...")
    start_pred_gen = time.time()
    prediction_samples = generate_realistic_prediction_samples(num_prediction_samples)
    end_pred_gen = time.time()
    print(f"Prediction samples generated in {end_pred_gen - start_pred_gen:.2f} seconds")

    # Prediction phase
    print(f"Predicting on {num_prediction_samples} random samples...")
    start_predict = time.time()
    predictions = method.predict(prediction_samples)
    end_predict = time.time()
    prediction_time = end_predict - start_predict

    # Results
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Prediction throughput: {num_prediction_samples / prediction_time:.2f} samples/second")

    # Validation
    assert (
        len(predictions) == num_prediction_samples
    ), f"Expected {num_prediction_samples} predictions, got {len(predictions)}"

    # Sample results analysis
    non_empty_predictions = [p for p in predictions if p.strip()]
    print(
        f"Non-empty predictions: {len(non_empty_predictions)}/{len(predictions)} ({len(non_empty_predictions)/len(predictions)*100:.1f}%)"
    )

    # Show some sample results
    print("\nSample prediction results (random data):")
    for i in range(min(10, len(prediction_samples))):
        original = " ".join(prediction_samples[i].get_input_text_by_lines())
        predicted = predictions[i]
        print(f"  Original: '{original}' -> Predicted: '{predicted}'")

    total_time = training_time + prediction_time
    print(f"\nTotal test time: {total_time:.2f} seconds")
    print(f"Overall throughput: {(num_training_samples + num_prediction_samples) / total_time:.2f} samples/second")

    print("Random data load test completed successfully!")


def test_regex_subtraction_method_scaling_random():
    """Test scaling behavior with different sample sizes using random data"""
    print("Testing RegexSubtractionMethod scaling behavior with random data...")

    identifier = ExtractionIdentifier(run_name="random_scaling_test", extraction_name="regex_random_scale")

    # Test different sizes - larger for stress testing
    test_sizes = [30000]

    results = {}

    for size in test_sizes:
        print(f"\nTesting with {size} random samples...")

        method = RegexSubtractionMethod(extraction_identifier=identifier)

        # Generate random data
        training_samples = generate_realistic_training_samples(min(size // 10, 5000))  # Cap training
        extraction_data = ExtractionData(samples=training_samples, extraction_identifier=identifier)

        # Train
        start_train = time.time()
        method.train(extraction_data)
        train_time = time.time() - start_train

        # Predict with random data
        prediction_samples = generate_realistic_prediction_samples(size)
        start_predict = time.time()
        predictions = method.predict(prediction_samples)
        predict_time = time.time() - start_predict

        throughput = size / predict_time
        results[size] = {
            "train_time": train_time,
            "predict_time": predict_time,
            "throughput": throughput,
            "predictions_count": len(predictions),
            "non_empty_predictions": len([p for p in predictions if p.strip()]),
        }

        print(f"  Training: {train_time:.2f}s, Prediction: {predict_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/second")
        print(
            f"  Non-empty predictions: {results[size]['non_empty_predictions']}/{len(predictions)} ({results[size]['non_empty_predictions']/len(predictions)*100:.1f}%)"
        )

    # Summary
    print("\nRandom data scaling test summary:")
    for size, result in results.items():
        print(
            f"  {size:,} samples: {result['throughput']:.2f} samples/sec | Non-empty: {result['non_empty_predictions']:,}/{result['predictions_count']:,} ({result['non_empty_predictions']/result['predictions_count']*100:.1f}%)"
        )

    print("Random data scaling test completed!")


if __name__ == "__main__":
    test_regex_subtraction_method_comprehensive_load()
    print("\n" + "=" * 80 + "\n")
    test_regex_subtraction_method_scaling_random()
