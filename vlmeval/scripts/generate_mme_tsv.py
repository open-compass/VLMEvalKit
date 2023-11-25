import os
import csv
import pathlib


DATASET_PATH = pathlib.Path("/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release")
OUTPUT_PATH = DATASET_PATH / "mme.tsv"

# each entry has the following properties:
# index, category, image_path, question, answer
dataset = []
index = 0


def add_entry(category, image_path, question, answer):
    global index
    dataset.append(
        {"index": index, "category": category, "image_path": image_path, "question": question, "answer": answer}
    )
    index += 1


def write_tsv():
    with open(OUTPUT_PATH, "w") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["index", "category", "image_path", "question", "answer"])
        writer.writeheader()
        for entry in dataset:
            writer.writerow(entry)


def parse_category(category):
    def mode1(category):
        folder = DATASET_PATH / category
        img_dir = folder / "images"
        qa_dir = folder / "questions_answers_YN"
        for txt in sorted(os.listdir(qa_dir)):
            name = txt.split(".")[0]
            if (img_dir / f"{name}.jpg").exists():
                img_type = "jpg"
            elif (img_dir / f"{name}.png").exists():
                img_type = "png"
            else:
                raise Exception(f"Image {name} not found in {img_dir}")
            with open(qa_dir / txt, "r") as f:
                lines = f.readlines()
            for line in lines:
                question, answer = line.split("\t")
                add_entry(category, f"{category}/images/{name}.{img_type}", question.strip(), answer.strip())

    def mode2(category):
        folder = DATASET_PATH / category
        for txt in sorted(os.listdir(folder)):
            if txt.endswith(".txt"):
                name = txt.split(".")[0]
                if (folder / f"{name}.jpg").exists():
                    img_type = "jpg"
                elif (folder / f"{name}.png").exists():
                    img_type = "png"
                else:
                    raise Exception(f"Image {name} not found in {folder}")
                with open(folder / txt, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    question, answer = line.split("\t")
                    add_entry(category, f"{category}/{name}.{img_type}", question.strip(), answer.strip())

    handlers = {
        "artwork": mode1,
        "celebrity": mode1,
        "code_reasoning": mode2,
        "color": mode2,
        "commonsense_reasoning": mode2,
        "count": mode2,
        "existence": mode2,
        "landmark": mode1,
        "numerical_calculation": mode2,
        "OCR": mode2,
        "position": mode2,
        "posters": mode1,
        "scene": mode1,
        "text_translation": mode2,
    }

    handlers[category](category)


if __name__ == "__main__":
    # list all folders under DATASET_PATH
    categories = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    for category in sorted(categories):
        print(f"Processing {category}...")
        parse_category(category)

    write_tsv()
