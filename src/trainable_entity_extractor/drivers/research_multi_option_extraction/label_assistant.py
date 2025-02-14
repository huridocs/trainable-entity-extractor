import typer


def main():
    person_name = typer.prompt("Filter string for PDFs")
    print(f"Seeing PDF {person_name}")


if __name__ == "__main__":
    typer.run(main)
