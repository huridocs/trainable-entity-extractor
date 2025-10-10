from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from trainable_entity_extractor.domain.ExtractionData import ExtractionData


class OptionDistribution(BaseModel):
    option_id: str
    option_label: str
    count: int
    percentage: float


class LanguageDistribution(BaseModel):
    language_iso: str
    count: int
    percentage: float


class TextLengthStats(BaseModel):
    min_length: int
    max_length: int
    avg_length: float
    median_length: float


class ExtractionDataSummary(BaseModel):
    total_samples: int
    total_options: int
    has_pdf_data: bool
    empty_pdfs_count: int = 0
    languages: list[LanguageDistribution] = Field(default_factory=list)
    option_distribution: list[OptionDistribution] = Field(default_factory=list)
    label_text_stats: Optional[TextLengthStats] = None
    source_text_stats: Optional[TextLengthStats] = None
    samples_with_values: int = 0

    @staticmethod
    def from_extraction_data(extraction_data: ExtractionData) -> "ExtractionDataSummary":
        total_samples = len(extraction_data.samples)
        total_options = len(extraction_data.options) if extraction_data.options else 0

        has_pdf_data = any(sample.pdf_data and sample.pdf_data.get_text() for sample in extraction_data.samples)
        empty_pdfs_count = 0

        if has_pdf_data:
            for sample in extraction_data.samples:
                if sample.pdf_data:
                    if not sample.pdf_data.get_text():
                        empty_pdfs_count += 1

        language_counter = Counter()
        for sample in extraction_data.samples:
            if sample.labeled_data and sample.labeled_data.language_iso:
                language_counter[sample.labeled_data.language_iso] += 1

        languages = [
            LanguageDistribution(language_iso=lang, count=count, percentage=round(count / total_samples * 100, 2))
            for lang, count in language_counter.most_common()
        ]

        option_counter = Counter()
        for sample in extraction_data.samples:
            if sample.labeled_data and sample.labeled_data.values:
                for value in sample.labeled_data.values:
                    option_counter[value.id] += 1

        option_distribution = []
        if extraction_data.options:
            for option in extraction_data.options:
                count = option_counter.get(option.id, 0)
                option_distribution.append(
                    OptionDistribution(
                        option_id=option.id,
                        option_label=option.label,
                        count=count,
                        percentage=round(count / total_samples * 100, 2) if total_samples > 0 else 0,
                    )
                )
            option_distribution = sorted(option_distribution, key=lambda x: x.count, reverse=True)[:30]

        label_text_lengths = []
        source_text_lengths = []
        samples_with_values = 0

        for sample in extraction_data.samples:
            if sample.labeled_data:
                if sample.labeled_data.label_text:
                    label_text_lengths.append(len(sample.labeled_data.label_text))
                if sample.labeled_data.source_text:
                    source_text_lengths.append(len(sample.labeled_data.source_text))
                if sample.labeled_data.values:
                    samples_with_values += 1

        label_text_stats = None
        if label_text_lengths:
            sorted_lengths = sorted(label_text_lengths)
            label_text_stats = TextLengthStats(
                min_length=min(label_text_lengths),
                max_length=max(label_text_lengths),
                avg_length=round(sum(label_text_lengths) / len(label_text_lengths), 2),
                median_length=sorted_lengths[len(sorted_lengths) // 2],
            )

        source_text_stats = None
        if source_text_lengths:
            sorted_lengths = sorted(source_text_lengths)
            source_text_stats = TextLengthStats(
                min_length=min(source_text_lengths),
                max_length=max(source_text_lengths),
                avg_length=round(sum(source_text_lengths) / len(source_text_lengths), 2),
                median_length=sorted_lengths[len(sorted_lengths) // 2],
            )

        return ExtractionDataSummary(
            total_samples=total_samples,
            total_options=total_options,
            has_pdf_data=has_pdf_data,
            empty_pdfs_count=empty_pdfs_count,
            languages=languages,
            option_distribution=option_distribution,
            label_text_stats=label_text_stats,
            source_text_stats=source_text_stats,
            samples_with_values=samples_with_values,
        )

    def to_report_string(self) -> str:
        lines = [
            "Data Summary",
            "=" * 80,
            f"Total Samples: {self.total_samples}",
        ]

        if self.total_options:
            lines.append(f"Total Options: {self.total_options}")

        if self.total_options and self.option_distribution:
            lines.append("\nOption Distribution:")
            for dist in self.option_distribution:
                lines.append(f"  - {dist.option_label} (id: {dist.option_id}): {dist.count} samples ({dist.percentage}%)")

        if self.samples_with_values > 0:
            percentage = round(self.samples_with_values / self.total_samples * 100, 2)
            lines.append(f"\nSamples with Option Values: {self.samples_with_values} ({percentage}%)")

        if self.languages:
            lines.append("\nLanguage Distribution:")
            for lang_dist in self.languages:
                lines.append(f"  - {lang_dist.language_iso}: {lang_dist.count} samples ({lang_dist.percentage}%)")

        if self.has_pdf_data:
            lines.append(f"\nPDF Data: Present")
            if self.empty_pdfs_count > 0:
                lines.append(f"Empty PDFs: {self.empty_pdfs_count}")

        if self.label_text_stats:
            lines.append("\nLabel Text Length:")
            lines.append(f"  - Min: {self.label_text_stats.min_length}")
            lines.append(f"  - Max: {self.label_text_stats.max_length}")
            lines.append(f"  - Average: {self.label_text_stats.avg_length}")
            lines.append(f"  - Median: {self.label_text_stats.median_length}")

        if self.source_text_stats:
            lines.append("\nSource Text Length:")
            lines.append(f"  - Min: {self.source_text_stats.min_length}")
            lines.append(f"  - Max: {self.source_text_stats.max_length}")
            lines.append(f"  - Average: {self.source_text_stats.avg_length}")
            lines.append(f"  - Median: {self.source_text_stats.median_length}")

        lines.append("=" * 80)

        return "\n".join(lines)
