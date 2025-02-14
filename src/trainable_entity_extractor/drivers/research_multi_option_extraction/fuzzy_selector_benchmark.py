from pdf_token_type_labels.TaskMistakes import TaskMistakes
from sklearn.metrics import f1_score

from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import (
    PdfMultiOptionMethod,
)
from trainable_entity_extractor.use_cases.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.use_cases.extractors.segment_selector.evaluate_config import PDF_LABELED_DATA_PATH

from scripts.pdf_multi_option_classification_benchmark import get_multi_option_benchmark_data


def show_mistakes(prediction_samples):
    for prediction_sample in prediction_samples:
        file_name = prediction_sample.pdf_data.pdf_features.file_type
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_PATH, "in_favor_segment_selector", file_name)
        for pdf_segment in prediction_sample.pdf_data.pdf_data_segments:
            if pdf_segment.ml_label:
                task_mistakes.add(pdf_segment.page_number, pdf_segment.bounding_box, 1, 1)
        task_mistakes.save()


def run():
    task_labeled_data = get_multi_option_benchmark_data(["countries_in_favor"])[0]

    values_list = [x.labeled_data.values for x in task_labeled_data.samples[20:]]
    truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, task_labeled_data.options)
    prediction_data = [PredictionSample(pdf_data=x.pdf_data) for x in task_labeled_data.samples[20:]]

    training_data = task_labeled_data
    training_data.samples = training_data.samples[:20]

    extractor = PdfToMultiOptionExtractor(task_labeled_data.extraction_identifier)
    # extractor.create_model(training_data)
    suggestions = extractor.get_suggestions(prediction_data)

    show_mistakes(prediction_data)

    values_list = [x.values for x in suggestions]
    predictions_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, task_labeled_data.options)

    performance = 100 * f1_score(truth_one_hot, predictions_one_hot, average="micro")

    print("performance")
    print(performance)


if __name__ == "__main__":
    run()
