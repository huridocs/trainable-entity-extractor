from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary
from trainable_entity_extractor.domain.TrainingSample import TrainingSample


class TestPerformanceSummary:

    def test_from_extraction_data_empty_samples(self):
        """Test creating PerformanceSummary from ExtractionData with no samples"""
        extraction_data = ExtractionData(samples=[])

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Test Extractor",
            training_samples_count=10,
            testing_samples_count=5,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Test Extractor"
        assert result.samples_count == 0
        assert result.options_count == 0
        assert result.languages == []
        assert result.training_samples_count == 10
        assert result.testing_samples_count == 5
        assert result.methods == []

    def test_from_extraction_data_with_samples_no_languages(self):
        """Test creating PerformanceSummary from ExtractionData with samples but no language info"""
        # Create samples without language_iso
        sample1 = TrainingSample(labeled_data=LabeledData(source_text="Sample 1"))
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Sample 2"))

        extraction_identifier = ExtractionIdentifier(run_name="test_run", extraction_name="test_extraction")
        extraction_data = ExtractionData(samples=[sample1, sample2], extraction_identifier=extraction_identifier)

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Multi Sample Extractor",
            training_samples_count=15,
            testing_samples_count=8,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Multi Sample Extractor"
        assert result.samples_count == 2
        assert result.options_count == 0
        assert result.languages == []
        assert result.training_samples_count == 15
        assert result.testing_samples_count == 8
        assert result.extraction_identifier == extraction_identifier

    def test_from_extraction_data_with_languages(self):
        """Test creating PerformanceSummary from ExtractionData with multiple languages"""
        # Create samples with different languages
        sample1 = TrainingSample(labeled_data=LabeledData(source_text="English text", language_iso="en"))
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Texto español", language_iso="es"))
        sample3 = TrainingSample(labeled_data=LabeledData(source_text="More English", language_iso="en"))
        sample4 = TrainingSample(labeled_data=LabeledData(source_text="Texte français", language_iso="fr"))

        extraction_data = ExtractionData(samples=[sample1, sample2, sample3, sample4])

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Multilingual Extractor",
            training_samples_count=20,
            testing_samples_count=12,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Multilingual Extractor"
        assert result.samples_count == 4
        assert result.options_count == 0
        assert set(result.languages) == {"en", "es", "fr"}
        assert result.training_samples_count == 20
        assert result.testing_samples_count == 12

    def test_from_extraction_data_with_options(self):
        """Test creating PerformanceSummary from ExtractionData with options"""
        option1 = Option(id="1", label="Option 1")
        option2 = Option(id="2", label="Option 2")
        option3 = Option(id="3", label="Option 3")

        sample = TrainingSample(labeled_data=LabeledData(source_text="Sample", language_iso="en"))
        extraction_data = ExtractionData(samples=[sample], options=[option1, option2, option3])

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Options Extractor",
            training_samples_count=5,
            testing_samples_count=3,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Options Extractor"
        assert result.samples_count == 1
        assert result.options_count == 3
        assert result.languages == ["en"]
        assert result.training_samples_count == 5
        assert result.testing_samples_count == 3

    def test_from_extraction_data_no_options_attribute(self):
        """Test creating PerformanceSummary from ExtractionData when options is None"""
        sample = TrainingSample(labeled_data=LabeledData(source_text="Sample", language_iso="de"))
        extraction_data = ExtractionData(samples=[sample])  # Don't explicitly set options=None

        result = PerformanceSummary.from_extraction_data(
            extractor_name="No Options Extractor",
            training_samples_count=7,
            testing_samples_count=4,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "No Options Extractor"
        assert result.samples_count == 1
        assert result.options_count == 0
        assert result.languages == ["de"]
        assert result.training_samples_count == 7
        assert result.testing_samples_count == 4

    def test_from_extraction_data_samples_without_labeled_data(self):
        """Test creating PerformanceSummary from ExtractionData with samples that have no labeled_data"""
        sample1 = TrainingSample()  # No labeled_data
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Valid sample", language_iso="pt"))

        extraction_data = ExtractionData(samples=[sample1, sample2])

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Mixed Data Extractor",
            training_samples_count=10,
            testing_samples_count=6,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Mixed Data Extractor"
        assert result.samples_count == 2
        assert result.options_count == 0
        assert result.languages == ["pt"]  # Only the valid sample with language is included
        assert result.training_samples_count == 10
        assert result.testing_samples_count == 6

    def test_from_extraction_data_duplicate_languages(self):
        """Test that duplicate languages are deduplicated"""
        sample1 = TrainingSample(labeled_data=LabeledData(source_text="First", language_iso="en"))
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Second", language_iso="en"))
        sample3 = TrainingSample(labeled_data=LabeledData(source_text="Third", language_iso="es"))
        sample4 = TrainingSample(labeled_data=LabeledData(source_text="Fourth", language_iso="en"))

        extraction_data = ExtractionData(samples=[sample1, sample2, sample3, sample4])

        result = PerformanceSummary.from_extraction_data(
            extractor_name="Duplicate Lang Extractor",
            training_samples_count=25,
            testing_samples_count=15,
            extraction_data=extraction_data,
        )

        assert result.extractor_name == "Duplicate Lang Extractor"
        assert result.samples_count == 4
        assert set(result.languages) == {"en", "es"}
        assert len(result.languages) == 2  # Duplicates removed

    def test_to_log_basic_summary_no_methods(self):
        """Test to_log with basic summary but no performance methods"""
        summary = PerformanceSummary(
            extractor_name="Basic Extractor",
            samples_count=50,
            options_count=0,
            languages=["en", "es"],
            training_samples_count=40,
            testing_samples_count=10,
            extraction_identifier=ExtractionIdentifier(run_name="run1", extraction_name="extract1"),
        )

        result = summary.to_log()

        assert "Performance summary" in result
        assert "Basic Extractor" in result
        assert "Best method: No methods - 10 mistakes / 0.00%" in result
        assert "Samples: 50" in result
        assert "Train/test: 40/10" in result
        assert "2 language(s): en, es" in result
        assert "Options count:" not in result  # Should not appear when options_count is 0
        assert "Methods by performance:" in result

    def test_to_log_with_options(self):
        """Test to_log when options_count > 0"""
        summary = PerformanceSummary(
            extractor_name="Options Extractor",
            samples_count=25,
            options_count=5,
            languages=["fr"],
            training_samples_count=20,
            testing_samples_count=5,
            extraction_identifier=ExtractionIdentifier(run_name="run2", extraction_name="extract2"),
        )

        result = summary.to_log()

        assert "Options Extractor" in result
        assert "1 language(s): fr" in result
        assert "Options count: 5" in result

    def test_to_log_no_languages(self):
        """Test to_log when there are no languages"""
        summary = PerformanceSummary(
            extractor_name="No Lang Extractor",
            samples_count=15,
            options_count=0,
            languages=[],
            training_samples_count=12,
            testing_samples_count=3,
        )

        result = summary.to_log()

        assert "0 language(s): None" in result

    def test_to_log_with_single_method(self):
        """Test to_log with a single performance method"""
        summary = PerformanceSummary(
            extractor_name="Single Method Extractor",
            samples_count=30,
            options_count=0,
            languages=["de"],
            training_samples_count=25,
            testing_samples_count=5,
        )
        summary.add_performance("Neural Network", 85.5)

        result = summary.to_log()

        assert "Best method: Neural Network - 1 mistake / 85.50%" in result
        assert "Methods by performance:" in result
        assert "Neural Network - 1 mistake / 85.50%" in result.split("Methods by performance:")[1]

    def test_to_log_with_multiple_methods_sorted(self):
        """Test to_log with multiple methods sorted by performance"""
        summary = PerformanceSummary(
            extractor_name="Multi Method Extractor",
            samples_count=40,
            options_count=3,
            languages=["en", "pt"],
            training_samples_count=30,
            testing_samples_count=10,
        )
        summary.add_performance("Method A", 75.0)
        summary.add_performance("Method B", 90.0)
        summary.add_performance("Method C", 60.0)

        result = summary.to_log()

        # Best method should be Method B (90.0%)
        assert "Best method: Method B - 1 mistake / 90.00%" in result

        # Methods should be sorted by performance (descending)
        methods_section = result.split("Methods by performance:")[1]
        method_lines = [line.strip() for line in methods_section.split("\n") if line.strip()]

        assert "Method B - 1 mistake / 90.00%" == method_lines[0]
        assert "Method A - 2 mistakes / 75.00%" == method_lines[1]
        assert "Method C - 4 mistakes / 60.00%" == method_lines[2]

    def test_to_log_perfect_performance(self):
        """Test to_log with 100% performance (0 mistakes)"""
        summary = PerformanceSummary(
            extractor_name="Perfect Extractor",
            samples_count=20,
            options_count=0,
            languages=["ja"],
            training_samples_count=15,
            testing_samples_count=5,
        )
        summary.add_performance("Perfect Method", 100.0)

        result = summary.to_log()

        assert "Best method: Perfect Method - 0 mistakes / 100.00%" in result
        assert "Perfect Method - 0 mistakes / 100.00%" in result

    def test_to_log_single_mistake(self):
        """Test to_log with exactly 1 mistake (singular form)"""
        summary = PerformanceSummary(
            extractor_name="Single Mistake Extractor",
            samples_count=25,
            options_count=0,
            languages=["it"],
            training_samples_count=20,
            testing_samples_count=10,
        )
        summary.add_performance("Almost Perfect", 90.0)  # 10% of 10 = 1 mistake

        result = summary.to_log()

        assert "Best method: Almost Perfect - 1 mistake / 90.00%" in result

    def test_to_log_many_languages(self):
        """Test to_log with multiple languages"""
        summary = PerformanceSummary(
            extractor_name="Multilingual Extractor",
            samples_count=100,
            options_count=8,
            languages=["en", "es", "fr", "de", "it", "pt"],
            training_samples_count=80,
            testing_samples_count=20,
        )
        summary.add_performance("Multilingual Model", 82.5)

        result = summary.to_log()

        assert "6 language(s): en, es, fr, de, it, pt" in result
        assert "Options count: 8" in result
        assert "Best method: Multilingual Model - 4 mistakes / 82.50%" in result

    def test_to_log_zero_performance(self):
        """Test to_log with 0% performance (all mistakes)"""
        summary = PerformanceSummary(
            extractor_name="Failed Extractor",
            samples_count=10,
            options_count=0,
            languages=["en"],
            training_samples_count=8,
            testing_samples_count=2,
        )
        summary.add_performance("Failed Method", 0.0)

        result = summary.to_log()

        assert "Best method: Failed Method - 2 mistakes / 0.00%" in result

    def test_from_extraction_data_and_to_log(self):
        """Test from_extraction_data and to_log integration"""
        # Prepare extraction data with languages and options
        from trainable_entity_extractor.domain.ExtractionData import ExtractionData
        from trainable_entity_extractor.domain.LabeledData import LabeledData
        from trainable_entity_extractor.domain.Option import Option
        from trainable_entity_extractor.domain.TrainingSample import TrainingSample

        sample1 = TrainingSample(labeled_data=LabeledData(source_text="Text 1", language_iso="en"))
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Text 2", language_iso="fr"))
        option1 = Option(id="1", label="Option 1")
        option2 = Option(id="2", label="Option 2")
        extraction_identifier = ExtractionIdentifier(run_name="test_run", extraction_name="test_extraction")
        extraction_data = ExtractionData(
            samples=[sample1, sample2], options=[option1, option2], extraction_identifier=extraction_identifier
        )

        summary = PerformanceSummary.from_extraction_data(
            extractor_name="Integration Extractor",
            training_samples_count=8,
            testing_samples_count=2,
            extraction_data=extraction_data,
        )
        summary.add_performance("test_method", 0.85)
        log = summary.to_log()
        assert "Integration Extractor" in log
        assert "2 language(s): en, fr" in log or "2 language(s): fr, en" in log
        assert "Options count: 2" in log
        assert "test_method" in log
        assert "0.85" in log

    def test_from_extraction_data_and_to_log_with_extraction_identifier(self):
        """Test from_extraction_data and to_log integration with ExtractionIdentifier"""
        from trainable_entity_extractor.domain.ExtractionData import ExtractionData
        from trainable_entity_extractor.domain.LabeledData import LabeledData
        from trainable_entity_extractor.domain.Option import Option
        from trainable_entity_extractor.domain.TrainingSample import TrainingSample
        from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier

        extraction_identifier = ExtractionIdentifier(run_name="run42", extraction_name="extract99")
        sample1 = TrainingSample(labeled_data=LabeledData(source_text="Text 1", language_iso="en"))
        sample2 = TrainingSample(labeled_data=LabeledData(source_text="Text 2", language_iso="fr"))
        option1 = Option(id="1", label="Option 1")
        option2 = Option(id="2", label="Option 2")
        extraction_data = ExtractionData(
            samples=[sample1, sample2], options=[option1, option2], extraction_identifier=extraction_identifier
        )

        summary = PerformanceSummary.from_extraction_data(
            extractor_name="Integration Extractor",
            training_samples_count=8,
            testing_samples_count=2,
            extraction_data=extraction_data,
        )
        summary.add_performance("test_method", 0.85)
        log = summary.to_log()
        assert "Integration Extractor" in log
        assert "2 language(s): en, fr" in log or "2 language(s): fr, en" in log
        assert "Options count: 2" in log
        assert "test_method" in log
        assert "0.85" in log
        assert "run42 / extract99" in log  # ExtractionIdentifier string representation
