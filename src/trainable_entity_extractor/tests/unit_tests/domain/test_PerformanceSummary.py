from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.PerformanceSummary import PerformanceSummary
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class TestPerformanceSummary:

    def test_direct_instantiation_empty(self):
        result = PerformanceSummary(
            extractor_name="Test Extractor",
            samples_count=0,
            options_count=0,
            languages=[],
            training_samples_count=10,
            testing_samples_count=5,
        )

        assert result.extractor_name == "Test Extractor"
        assert result.samples_count == 0
        assert result.options_count == 0
        assert result.languages == []
        assert result.training_samples_count == 10
        assert result.testing_samples_count == 5
        assert result.performances == []

    def test_direct_instantiation_with_extraction_identifier(self):
        extraction_identifier = ExtractionIdentifier(run_name="test_run", extraction_name="test_extraction")

        result = PerformanceSummary(
            extractor_name="Multi Sample Extractor",
            samples_count=2,
            options_count=0,
            languages=[],
            training_samples_count=15,
            testing_samples_count=8,
            extraction_identifier=extraction_identifier,
        )

        assert result.extractor_name == "Multi Sample Extractor"
        assert result.samples_count == 2
        assert result.options_count == 0
        assert result.languages == []
        assert result.training_samples_count == 15
        assert result.testing_samples_count == 8
        assert result.extraction_identifier == extraction_identifier

    def test_direct_instantiation_with_languages(self):
        result = PerformanceSummary(
            extractor_name="Multilingual Extractor",
            samples_count=4,
            options_count=0,
            languages=["en", "es", "fr"],
            training_samples_count=20,
            testing_samples_count=12,
        )

        assert result.extractor_name == "Multilingual Extractor"
        assert result.samples_count == 4
        assert result.options_count == 0
        assert set(result.languages) == {"en", "es", "fr"}
        assert result.training_samples_count == 20
        assert result.testing_samples_count == 12

    def test_direct_instantiation_with_options(self):
        result = PerformanceSummary(
            extractor_name="Options Extractor",
            samples_count=1,
            options_count=3,
            languages=["en"],
            training_samples_count=5,
            testing_samples_count=3,
        )

        assert result.extractor_name == "Options Extractor"
        assert result.samples_count == 1
        assert result.options_count == 3
        assert result.languages == ["en"]
        assert result.training_samples_count == 5
        assert result.testing_samples_count == 3

    def test_direct_instantiation_no_options(self):
        result = PerformanceSummary(
            extractor_name="No Options Extractor",
            samples_count=1,
            options_count=0,
            languages=["de"],
            training_samples_count=7,
            testing_samples_count=4,
        )

        assert result.extractor_name == "No Options Extractor"
        assert result.samples_count == 1
        assert result.options_count == 0
        assert result.languages == ["de"]
        assert result.training_samples_count == 7
        assert result.testing_samples_count == 4

    def test_direct_instantiation_mixed_data(self):
        result = PerformanceSummary(
            extractor_name="Mixed Data Extractor",
            samples_count=2,
            options_count=0,
            languages=["pt"],
            training_samples_count=10,
            testing_samples_count=6,
        )

        assert result.extractor_name == "Mixed Data Extractor"
        assert result.samples_count == 2
        assert result.options_count == 0
        assert result.languages == ["pt"]
        assert result.training_samples_count == 10
        assert result.testing_samples_count == 6

    def test_direct_instantiation_with_empty_pdf_count(self):
        result = PerformanceSummary(
            extractor_name="PDF Extractor",
            samples_count=4,
            options_count=0,
            languages=["en", "es"],
            training_samples_count=25,
            testing_samples_count=15,
            empty_pdf_count=2,
        )

        assert result.extractor_name == "PDF Extractor"
        assert result.samples_count == 4
        assert result.options_count == 0
        assert set(result.languages) == {"en", "es"}
        assert result.empty_pdf_count == 2

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
        assert "Best method: No methods - 0s / 10 mistakes / 0.00%" in result
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

        assert "Best method: Neural Network - 0s / 1 mistake / 85.50%" in result
        assert "Methods by performance:" in result
        assert "Neural Network - 0s / 1 mistake / 85.50%" in result.split("Methods by performance:")[1]

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
        assert "Best method: Method B - 0s / 1 mistake / 90.00%" in result

        # Check that methods are sorted by performance (highest first)
        methods_section = result.split("Methods by performance:")[1]
        assert "Method B - 0s / 1 mistake / 90.00%" in methods_section
        assert "Method A - 0s / 2 mistakes / 75.00%" in methods_section
        assert "Method C - 0s / 4 mistakes / 60.00%" in methods_section

        # Verify order - Method B should appear before Method A, which should appear before Method C
        method_b_pos = methods_section.find("Method B")
        method_a_pos = methods_section.find("Method A")
        method_c_pos = methods_section.find("Method C")
        assert method_b_pos < method_a_pos < method_c_pos

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

        assert "Best method: Perfect Method - 0s / 0 mistakes / 100.00%" in result
        assert "Perfect Method - 0s / 0 mistakes / 100.00%" in result

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

        assert "Best method: Almost Perfect - 0s / 1 mistake / 90.00%" in result
        assert "Almost Perfect - 0s / 1 mistake / 90.00%" in result

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
        assert "Best method: Multilingual Model - 0s / 4 mistakes / 82.50%" in result

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

        assert "Best method: Failed Method - 0s / 2 mistakes / 0.00%" in result
        assert "Failed Method - 0s / 2 mistakes / 0.00%" in result

    def test_add_performance_and_to_log(self):
        extraction_identifier = ExtractionIdentifier(run_name="test_run", extraction_name="test_extraction")

        summary = PerformanceSummary(
            extractor_name="Integration Extractor",
            samples_count=2,
            options_count=2,
            languages=["en", "fr"],
            training_samples_count=8,
            testing_samples_count=2,
            extraction_identifier=extraction_identifier,
        )
        summary.add_performance("test_method", 85.0)
        log = summary.to_log()
        assert "Integration Extractor" in log
        assert "2 language(s): en, fr" in log or "2 language(s): fr, en" in log
        assert "Options count: 2" in log
        assert "test_method" in log
        assert "85.00%" in log

    def test_add_performance_and_to_log_with_extraction_identifier(self):
        extraction_identifier = ExtractionIdentifier(run_name="run42", extraction_name="extract99")

        summary = PerformanceSummary(
            extractor_name="Integration Extractor",
            samples_count=2,
            options_count=2,
            languages=["en", "fr"],
            training_samples_count=8,
            testing_samples_count=2,
            extraction_identifier=extraction_identifier,
        )
        summary.add_performance("test_method", 85.0)
        log = summary.to_log()
        assert "Integration Extractor" in log
        assert "2 language(s): en, fr" in log or "2 language(s): fr, en" in log
        assert "Options count: 2" in log
        assert "test_method" in log
        assert "85.00%" in log
        assert "run42 / extract99" in log

    def test_from_distributed_job(self):
        extraction_identifier = ExtractionIdentifier(run_name="dist_run", extraction_name="dist_extraction")

        extractor_job = TrainableEntityExtractorJob(
            run_name="dist_run",
            extraction_name="dist_extraction",
            extractor_name="Test Extractor",
            method_name="test_method",
            options=[],
            gpu_needed=False,
            timeout=300,
        )

        performance_result = Performance(
            performance=90.0,
            testing_samples_count=10,
            training_samples_count=40,
            samples_count=50,
        )

        sub_job = DistributedSubJob(
            extractor_job=extractor_job,
            status=JobStatus.SUCCESS,
            result=performance_result,
        )

        distributed_job = DistributedJob(
            type=JobType.PERFORMANCE,
            sub_jobs=[sub_job],
            extraction_identifier=extraction_identifier,
        )

        summary = PerformanceSummary.from_distributed_job(distributed_job)

        assert summary.extractor_name == "Test Extractor"
        assert summary.samples_count == 50
        assert summary.options_count == 0
        assert summary.training_samples_count == 40
        assert summary.testing_samples_count == 10
        assert summary.extraction_identifier == extraction_identifier
        assert summary.languages == []

    def test_add_performance_from_sub_job(self):
        summary = PerformanceSummary(
            extractor_name="Test Extractor",
            samples_count=10,
            options_count=0,
            languages=["en"],
            training_samples_count=8,
            testing_samples_count=2,
        )

        extractor_job = TrainableEntityExtractorJob(
            run_name="test_run",
            extraction_name="test_extraction",
            extractor_name="Test Extractor",
            method_name="method_a",
            options=[],
            gpu_needed=False,
            timeout=300,
        )

        performance_result = Performance(
            performance=85.5,
            testing_samples_count=2,
            training_samples_count=8,
            samples_count=10,
        )

        sub_job = DistributedSubJob(
            extractor_job=extractor_job,
            status=JobStatus.SUCCESS,
            result=performance_result,
        )

        summary.add_performance_from_sub_job(sub_job)

        assert len(summary.performances) == 1
        assert summary.performances[0].method_name == "method_a"
        assert summary.performances[0].performance == 85.5
        assert summary.performances[0].failed == False
