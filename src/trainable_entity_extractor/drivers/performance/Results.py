from os.path import dirname, realpath, join
import time
from datetime import datetime
from rich import box
from rich.table import Table
from rich.console import Console

SCRIPT_PATH = dirname(realpath(__file__))


class Results:
    RESULTS_PATH = "results"

    def __init__(self, results_name: str):
        self.method = ""
        self.start_time = None
        self.results_name = results_name
        self.results_path = join(SCRIPT_PATH, "..", "..", "performance_results", f"{results_name}.md")
        self.table = self.initiate_table()
        self.accuracies = list()

    @staticmethod
    def initiate_table():
        table = Table(title="", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Dataset")
        table.add_column("Method")
        table.add_column("Train \n size", justify="right")
        table.add_column("Test \n size", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Acc.", justify="right")
        return table

    def set_start_time(self):
        print("start time")
        self.start_time = time.time()

    def get_total_time(self):
        total_time = round(time.time() - self.start_time)
        print(f"{datetime.now():%Y_%m_%d_%H_%M}", f"finished in {total_time} seconds")
        return str(total_time) + "s"

    @staticmethod
    def format_dataset_name(name):
        return name.replace(".tsv", "").replace("_", " ")

    def save_result(self, dataset: str, method: str, accuracy: float, train_length: int, test_length: int):
        self.accuracies.append(accuracy)
        self.table.add_row(
            self.format_dataset_name(dataset),
            method,
            str(train_length),
            str(test_length),
            self.get_total_time(),
            str(round(accuracy)) + "%",
        )

    def write_results(self):
        accuracies_average = round(sum(self.accuracies) / len(self.accuracies))
        self.table.add_row("", "", "", "", "", "")
        self.table.add_row("average", "", "", "", "", str(accuracies_average) + "%")
        console = Console(record=True)
        console.print(self.table)
        console.save_text(self.results_path)
