import shutil
from pathlib import Path

METHOD_TO_COPY = "base_frequent_words"
NEW_METHOD_NAME = "titles_history"

METHODS_FOLDER = Path("methods").resolve()


def to_pascal_case(snake_str):
    return "".join([x.title() for x in snake_str.split("_")])


def replace_string_in_file(file_path: str, old_text: str, new_text: str):
    fin = open(file_path, "rt")
    data = fin.read()
    data = data.replace(old_text, new_text)
    fin.close()
    fin = open(file_path, "wt")
    fin.write(data)
    fin.close()


def copy_method():
    old_folder = f"{METHODS_FOLDER}/{METHOD_TO_COPY}"
    new_folder = f"{METHODS_FOLDER}/{NEW_METHOD_NAME}"

    print(old_folder, "old_folder")
    print(new_folder, "new_folder")

    old_class_name = to_pascal_case(METHOD_TO_COPY)
    new_class_name = to_pascal_case(NEW_METHOD_NAME)

    print(old_class_name, "old_class_name")
    print(new_class_name, "new_class_name")

    shutil.copytree(old_folder, new_folder)

    shutil.move(f"{new_folder}/{old_class_name}.py", f"{new_folder}/{new_class_name}.py")
    shutil.move(f"{new_folder}/Segment{old_class_name}.py", f"{new_folder}/Segment{new_class_name}.py")
    replace_string_in_file(f"{new_folder}/{new_class_name}.py", old_class_name, new_class_name)
    replace_string_in_file(f"{new_folder}/{new_class_name}.py", METHOD_TO_COPY, NEW_METHOD_NAME)
    replace_string_in_file(f"{new_folder}/Segment{new_class_name}.py", old_class_name, new_class_name)


if __name__ == "__main__":
    copy_method()
