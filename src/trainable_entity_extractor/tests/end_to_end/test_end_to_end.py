import shutil

from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample


def test_text_to_multi_option_fuzzy_all_100_perfect_accuracy():
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
        TrainingSample(labeled_data=LabeledData(source_text="I like banana and orange", values=[options[1], options[2]])),
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

    # Cleanup
    shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
