import shutil
from unittest import TestCase

from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.domain.PdfData import PdfData


class TestEndToEnd(TestCase):
    def test_text_to_multi_option_fuzzy_all_100_perfect_accuracy(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="test_tenant_test_user_test_doc_type_test_entity")

        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

        options = [
            Option(id="1", label="apple"),
            Option(id="2", label="banana"),
            Option(id="3", label="orange"),
        ]

        training_samples = [
            TrainingSample(labeled_data=LabeledData(source_text="I like apple", values=[options[0]])),
            TrainingSample(labeled_data=LabeledData(source_text="I like banana", values=[options[1]])),
            TrainingSample(labeled_data=LabeledData(source_text="I like orange", values=[options[2]])),
            TrainingSample(labeled_data=LabeledData(source_text="I like apple and banana", values=[options[0], options[1]])),
            TrainingSample(
                labeled_data=LabeledData(source_text="I like banana and orange", values=[options[1], options[2]])
            ),
            TrainingSample(labeled_data=LabeledData(source_text="I like apple and orange", values=[options[0], options[2]])),
            TrainingSample(
                labeled_data=LabeledData(
                    source_text="I like apple, banana and orange", values=[options[0], options[1], options[2]]
                )
            ),
            TrainingSample(labeled_data=LabeledData(source_text="I like nothing", values=[])),
        ]

        extraction_data = ExtractionData(
            extraction_identifier=extraction_identifier,
            options=options,
            samples=training_samples,
            multi_value=True,
        )

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)

        # Act
        success, message = trainable_entity_extractor.train(extraction_data)
        assert success, f"Training failed: {message}"

        # check that the best model is TextFuzzyAll100
        job = trainable_entity_extractor.model_storage.get_extractor_job(extraction_identifier)
        assert job.method_name == "TextFuzzyAll100"

        prediction_samples = [
            PredictionSample(entity_name="test_entity", source_text="I want an apple"),
            PredictionSample(entity_name="test_entity", source_text="I want a banana and an orange"),
        ]
        suggestions = trainable_entity_extractor.predict(prediction_samples)

        # Assert
        assert len(suggestions) == 2

        assert len(suggestions[0].values) == 1
        assert suggestions[0].values[0].id == "1"
        assert suggestions[0].values[0].label == "apple"

        assert len(suggestions[1].values) == 2
        suggestion_labels = {v.label for v in suggestions[1].values}
        assert suggestion_labels == {"banana", "orange"}

        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def test_pdf_to_multi_option(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="test_tenant_test_user_test_doc_type_test_entity_pdf")

        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

        options = [
            Option(id="1", label="apple"),
            Option(id="2", label="banana"),
            Option(id="3", label="orange"),
        ]

        training_samples = [
            TrainingSample(
                pdf_data=PdfData.from_texts(
                    ["apple, banana and orange", "some text", "some other text", "fruit is", "apple"]
                ),
                labeled_data=LabeledData(values=[options[0]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(
                    ["apple, banana and orange", "some text", "some other text", "fruit is", "banana"]
                ),
                labeled_data=LabeledData(values=[options[1]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(
                    ["apple, banana and orange", "some text", "some other text", "fruit is", "orange"]
                ),
                labeled_data=LabeledData(values=[options[2]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(
                    ["apple, banana and orange", "some text", "some other text", "fruit is", "apple and banana"]
                ),
                labeled_data=LabeledData(values=[options[0], options[1]]),
            ),
            TrainingSample(pdf_data=PdfData.from_texts(["I like nothing"]), labeled_data=LabeledData(values=[])),
        ]

        extraction_data = ExtractionData(
            extraction_identifier=extraction_identifier,
            options=options,
            samples=training_samples,
            multi_value=True,
        )

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)

        # Act
        success, message = trainable_entity_extractor.train(extraction_data)
        assert success, f"Training failed: {message}"

        # check that the best model is PreviousWordsTokenSelectorFuzzy75
        job = trainable_entity_extractor.model_storage.get_extractor_job(extraction_identifier)
        assert job.method_name == "PreviousWordsTokenSelectorFuzzy75"

        prediction_samples = [
            PredictionSample(
                entity_name="test_entity", pdf_data=PdfData.from_texts(["some text", "some other text", "fruit is", "apple"])
            ),
            PredictionSample(
                entity_name="test_entity",
                pdf_data=PdfData.from_texts(["some text", "some other text", "fruit is", "banana and orange"]),
            ),
        ]
        suggestions = trainable_entity_extractor.predict(prediction_samples)

        # Assert
        assert len(suggestions) == 2

        assert len(suggestions[0].values) == 1
        assert suggestions[0].values[0].id == "1"
        assert suggestions[0].values[0].label == "apple"

        assert len(suggestions[1].values) == 2
        suggestion_labels = {v.label for v in suggestions[1].values}
        assert suggestion_labels == {"banana", "orange"}

        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

    def test_pdf_to_multi_option_fuzzy_all_75_perfect_accuracy(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="test_tenant_test_user_test_doc_type_test_entity_pdf")

        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)

        options = [
            Option(id="1", label="The quick brown fox jumps over the lazy dog"),
            Option(id="2", label="My favorite programming language is Python"),
            Option(id="3", label="The capital of France is the city of Paris"),
        ]

        training_samples = [
            TrainingSample(
                pdf_data=PdfData.from_texts(["The quik brown fox jump over a lazy dog"]),
                labeled_data=LabeledData(values=[options[0]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(["My favorit programing language is Pithon"]),
                labeled_data=LabeledData(values=[options[1]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(["The capital of France is the city of Parris"]),
                labeled_data=LabeledData(values=[options[2]]),
            ),
            TrainingSample(
                pdf_data=PdfData.from_texts(
                    ["The quik brown fox jump over a lazy dog and My favorit programing language is Pithon"]
                ),
                labeled_data=LabeledData(values=[options[0], options[1]]),
            ),
            TrainingSample(pdf_data=PdfData.from_texts(["I like nothing"]), labeled_data=LabeledData(values=[])),
        ]

        extraction_data = ExtractionData(
            extraction_identifier=extraction_identifier,
            options=options,
            samples=training_samples,
            multi_value=True,
        )

        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)

        # Act
        success, message = trainable_entity_extractor.train(extraction_data)
        assert success, f"Training failed: {message}"

        # check that the best model is FuzzyAll75
        job = trainable_entity_extractor.model_storage.get_extractor_job(extraction_identifier)
        assert job.method_name == "FuzzyAll75"

        prediction_samples = [
            PredictionSample(
                entity_name="test_entity", pdf_data=PdfData.from_texts(["The quik brown fox jump over a lazy dog"])
            ),
            PredictionSample(
                entity_name="test_entity",
                pdf_data=PdfData.from_texts(
                    ["My favorit programing language is Pithon and The capital of France is the city of Parris"]
                ),
            ),
        ]
        suggestions = trainable_entity_extractor.predict(prediction_samples)

        # Assert
        assert len(suggestions) == 2

        assert len(suggestions[0].values) == 1
        assert suggestions[0].values[0].id == "1"
        assert suggestions[0].values[0].label == "The quick brown fox jumps over the lazy dog"

        assert len(suggestions[1].values) == 2
        suggestion_labels = {v.label for v in suggestions[1].values}
        assert suggestion_labels == {
            "My favorite programming language is Python",
            "The capital of France is the city of Paris",
        }

        # Cleanup
        shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
