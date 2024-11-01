<h3 align="center">Trainable Entity Extractor</h3>
<p align="center">Python Machine Learning Library for Self-Training Text Extractors</p>


This project provides a versatile Python library designed to self-train text extractors. 
It intelligently determines whether simple heuristics or a medium-size Language Model (LLM) 
is best suited for a given extraction task. This adaptive approach ensures optimal 
performance across a range of text extraction scenarios.

Supported Extraction Types:

* Text-to-Text: Extract specific text from a given input text.
* Text-to-Multi-Label: Classify input text into multiple relevant topics or categories.
* PDF-to-Multi-Label: Extract key information from PDFs and classify them into multiple categories.
* PDF-to-Text: Extract raw text content from PDF documents.


### Why Self-Training?

Zero-shot LLMs, while powerful, may struggle with certain trivial extraction tasks. 
Our self-training approach leverages the strengths of both rule-based and machine learning 
techniques to achieve superior performance and accuracy.


### Quick start

First install the library:

    pip install git+https://github.com/huridocs/trainable-entity-extractor


For Text-to-Text you can use the library as follows:


```py
extraction_identifier = ExtractionIdentifier(extraction_name="quick_start", output_path="./output")

training_samples = [TrainingSample.from_text(source_text="one 1", label_text="1", language_iso="en")]
training_samples += [TrainingSample.from_text(source_text="two 2", label_text="2", language_iso="en")]

extraction_data = ExtractionData(samples=training_samples, extraction_identifier=extraction_identifier)

trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
trainable_entity_extractor.train(extraction_data)

predictions_samples = [PredictionSample.from_text("test 0")]
predictions_samples += [PredictionSample.from_text("test 1")]

suggestions = trainable_entity_extractor.predict(predictions_samples)
# suggestions[0].text -> "0"
# suggestions[1].text ->"1"
```

For Text-to-Multi-Label you can use the library as follows:

    extraction_identifier = ExtractionIdentifier(extraction_name="quick_start", output_path="./output")

    options = [Option(id="0", label="0"), Option(id="1", label="1"), Option(id="2", label="2")] 

    training_samples = [TrainingSample(labeled_data=LabeledData(values=[options[1]], language_iso="en", source_text="one 1"))]
    training_samples += [TrainingSample(labeled_data=LabeledData(values=[options[2]], language_iso="en", source_text="two 2"))]
    extraction_data = ExtractionData(samples=training_samples, 
                                        extraction_identifier=extraction_identifier,
                                        options=options,
                                        multi_value=True)
    
    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier=extraction_identifier)
    trainable_entity_extractor.train(extraction_data)
    
    predictions_samples = [PredictionSample.from_text("test 0")]
    predictions_samples += [PredictionSample.from_text("test 1")]
    
    suggestions = trainable_entity_extractor.predict(predictions_samples)
    # suggestions[0].values -> [Option(id="0", label="0")
    # suggestions[1].values -> [Option(id="1", label="1")


### Technical details

Methods implemented for Text-to-Text extraction are:

* DateParserMethod
* GlinerDateParserMethod
* InputWithoutSpaces
* MT5TrueCaseEnglishSpanishMethod
* NerFirstAppearanceMethod
* NerLastAppearanceMethod
* RegexMethod
* RegexSubtractionMethodSameInputOutputMethod


Methods implemented for Text-to-Multi-Label extraction are:

* NaiveTextToMultiOptionMethod
* TextBert
* TextBertMultilingual
* TextFastTextMethod
* TextFuzzyAll75
* TextFuzzyAll88
* TextFuzzyAll100
* TextFuzzyFirst
* TextFuzzyFirstCleanLabels
* TextFuzzyLast
* TextFuzzyLastCleanLabels
* TextSetFit
* TextSetFitMultilingual
* TextSingleLabelBert
* TextSingleLabelSetFit
* TextSingleLabelSetFitMultilingual
* TextTfIdf

# Execute tests
make test
